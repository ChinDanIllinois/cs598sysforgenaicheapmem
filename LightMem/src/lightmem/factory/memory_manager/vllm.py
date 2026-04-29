
import concurrent
import json
import os
import time
from openai import OpenAI
from typing import List, Dict, Optional, Literal, Any, Union

from lightmem.configs.memory_manager.base_config import BaseMemoryManagerConfig
from lightmem.memory.utils import clean_response
from lightmem.memory.prompts import EXTRACTION_PROMPTS, METADATA_GENERATE_PROMPT


from .batch_manager import BatchManager


class VllmManager:
    def __init__(self, config: BaseMemoryManagerConfig, batch_manager: Optional[BatchManager] = None):
        self.config = config
        self.batch_manager = batch_manager

        if not self.config.model:
            raise ValueError("vLLM model is not specified. Refer to https://vllm.ai/docs/models/ for available models.")

        self.api_key = self.config.api_key or os.getenv("VLLM_API_KEY") or "EMPTY"
        self.base_url = (
            self.config.vllm_base_url
            or os.getenv("VLLM_BASE_URL")
            or "http://localhost:8000/v1"
        )

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=120)

    def _parse_response(self, response, tools):
        """
        Process the response based on whether tools are used or not.

        Args:
            response: The raw response from API.
            tools: The list of tools provided in the request.

        Returns:
            str or dict: The processed response.
        """
        if tools:
            processed_response = {
                "content": response.choices[0].message.content,
                "tool_calls": [],
            }

            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    processed_response["tool_calls"].append(
                        {
                            "name": tool_call.function.name,
                            "arguments": json.loads(tool_call.function.arguments),
                        }
                    )

            return processed_response
        else:
            return response.choices[0].message.content

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, str]] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ) -> tuple[Optional[str], Dict[str, Any]]:
        """
        Generate a response based on the given messages.

        Args:
            messages (list): List of message dicts containing 'role' and 'content'.
            response_format (dict, optional): Format of the response. Defaults to None.
            tools (list, optional): List of tools that the model can call. Defaults to None.
            tool_choice (str, optional): Tool choice method. Defaults to "auto".

        Returns:
            str: The generated response.
            dict: Usage info regarding tokens.
        """
        params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens // 2,  # Cap at half of config to prevent model throttling
            "top_p": self.config.top_p,
            "frequency_penalty": 1.2,  # Added to prevent small models from endless repetitive JSON generation
            "presence_penalty": 0.5,
        }

        if response_format:
            params["response_format"] = response_format
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        # Add any additional vLLM-specific parameters passed in kwargs
        params.update(kwargs)

        try:
            start_time = time.perf_counter()
            response = self.client.chat.completions.create(**params)
            time_taken = time.perf_counter() - start_time
        except Exception as e:
            print(f"Error calling vLLM API: {e}")
            return None, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "time_taken": 0.0}
            
        usage_info = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "time_taken": time_taken,
        }
        parsed_response = self._parse_response(response, tools)

        return parsed_response, usage_info

    def meta_text_extract(
        self,
        extract_list: List[List[List[Dict]]],
        messages_use: Literal["user_only", "assistant_only", "hybrid"] = "user_only",
        topic_id_mapping: Optional[List[List[int]]] = None,
        extraction_mode: Literal["flat", "event"] = "flat",
        custom_prompts: Optional[Dict[str, str]] = None  
    ) -> List[Optional[Dict]]:
        """
        Extract metadata from text segments using parallel processing.

        Args:
            extract_list: List of message segments to process
            messages_use: Strategy for which messages to use
            topic_id_mapping: For each API call, the global topic IDs
            extraction_mode: "flat" or "event"
            custom_prompts: Optional custom prompts. If None, use defaults from EXTRACTION_PROMPTS

        Returns:
            List of extracted metadata results, None for failed segments
        """
        if not extract_list:
            return []
        
        default_prompts = EXTRACTION_PROMPTS.get(extraction_mode, {})
        
        if custom_prompts is None:
            prompts = default_prompts
        else:
            prompts = {**default_prompts, **custom_prompts}
        
        if extraction_mode == "flat":
            return self._extract_with_prompt(
                system_prompt=prompts.get("factual", METADATA_GENERATE_PROMPT),
                extract_list=extract_list,
                messages_use=messages_use,
                topic_id_mapping=topic_id_mapping,
                entry_type="factual"
            )
        
        elif extraction_mode == "event":
            factual_results = self._extract_with_prompt(
                system_prompt=prompts["factual"],
                extract_list=extract_list,
                messages_use=messages_use,
                topic_id_mapping=topic_id_mapping,
                entry_type="factual"
            )
            
            relational_results = self._extract_with_prompt(
                system_prompt=prompts["relational"],
                extract_list=extract_list,
                messages_use=messages_use,
                topic_id_mapping=topic_id_mapping,
                entry_type="relational"
            )
            
            return self._merge_dual_perspective_results(
                factual_results, 
                relational_results
            )
        
        else:
            raise ValueError(f"Unknown extraction_mode: {extraction_mode}")
    
    def _merge_dual_perspective_results(
        self,
        factual_results: List[Optional[Dict]],
        relational_results: List[Optional[Dict]]
    ) -> List[Optional[Dict]]:
        """
        Args:
            factual_results: Factual extraction results
            relational_results: Relational extraction results
        
        Returns:
            Merged results with combined cleaned_result and accumulated usage
        """
        merged_results = []
        
        for factual, relational in zip(factual_results, relational_results):
            if factual is None and relational is None:
                merged_results.append(None)
                continue
            
            merged = {
                "input_prompt": [],
                "output_prompt": "",
                "cleaned_result": [],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }
            
            if factual is not None:
                merged["input_prompt"].extend(factual.get("input_prompt", []))
                merged["cleaned_result"].extend(factual.get("cleaned_result", []))
                if factual.get("usage"):
                    for key in merged["usage"]:
                        merged["usage"][key] += factual["usage"].get(key, 0)
            
            if relational is not None:
                merged["input_prompt"].extend(relational.get("input_prompt", []))
                merged["cleaned_result"].extend(relational.get("cleaned_result", []))
                if relational.get("usage"):
                    for key in merged["usage"]:
                        merged["usage"][key] += relational["usage"].get(key, 0)
            
            merged["output_prompt"] = (
                f"Factual: {factual.get('output_prompt', 'N/A') if factual else 'N/A'}\n"
                f"Relational: {relational.get('output_prompt', 'N/A') if relational else 'N/A'}"
            )
            
            merged_results.append(merged)
        
        return merged_results

    def _extract_with_prompt(
        self,
        system_prompt: str,
        extract_list: List[List[List[Dict]]],
        messages_use: str,
        topic_id_mapping: Optional[List[List[int]]],
        entry_type: str = "factual"
    ) -> List[Optional[Dict]]:
        """
        Args:
            system_prompt: System prompt for extraction
            extract_list: List of message segments
            messages_use: Message filtering strategy
            topic_id_mapping: Global topic IDs
            entry_type: "factual" or "relational"
        
        Returns:
            List of extraction results
        """
        def concatenate_messages(segment: List[Dict], messages_use: str) -> str:
            """Concatenate messages based on usage strategy"""
            role_filter = {
                "user_only": {"user"},
                "assistant_only": {"assistant"},
                "hybrid": {"user", "assistant"}
            }

            if messages_use not in role_filter:
                raise ValueError(f"Invalid messages_use value: {messages_use}")

            allowed_roles = role_filter[messages_use]
            message_lines = []

            for mes in segment:
                if mes.get("role") in allowed_roles:
                    sequence_id = mes["sequence_number"]
                    role = mes["role"]
                    content = mes.get("content", "")
                    speaker_name = mes.get("speaker_name", "")
                    time_stamp = mes.get("time_stamp", "")
                    weekday = mes.get("weekday", "")
                    
                    time_prefix = ""
                    if time_stamp and weekday:
                        time_prefix = f"[{time_stamp}, {weekday}] "

                    if speaker_name:
                        message_lines.append(f"{time_prefix}{sequence_id//2}.{speaker_name}: {content}")
                    else:
                        message_lines.append(f"{time_prefix}{sequence_id//2}.{role}: {content}")
            
            return "\n".join(message_lines)

        def get_metadata_messages(api_call_idx, api_call_segments):
            user_prompt_parts: List[str] = []
            
            global_topic_ids: List[int] = []
            if topic_id_mapping and api_call_idx < len(topic_id_mapping):
                global_topic_ids = topic_id_mapping[api_call_idx]

            for topic_idx, topic_segment in enumerate(api_call_segments):
                if topic_idx < len(global_topic_ids):
                    global_topic_id = global_topic_ids[topic_idx]
                else:
                    global_topic_id = topic_idx + 1
                
                topic_text = concatenate_messages(topic_segment, messages_use)
                user_prompt_parts.append(f"--- Topic {global_topic_id} ---\n{topic_text}")

            user_prompt = "\n".join(user_prompt_parts)
            
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

        if self.batch_manager:
            futures = []
            for idx, segments in enumerate(extract_list):
                metadata_messages = get_metadata_messages(idx, segments)
                # Batch processor expects (messages, config)
                # We pass response_format in config
                fut = self.batch_manager.add_request(
                    messages=metadata_messages,
                    config={"response_format": {"type": "json_object"}}
                )
                futures.append(fut)
            
            # Wait for all futures to complete
            results = []
            for idx, fut in enumerate(futures):
                try:
                    raw_response, usage_info = fut.result()
                    if raw_response is None:
                        raise ValueError("API returned None response due to timeout or error.")
                    metadata_facts = clean_response(raw_response)
                    for entry in metadata_facts:
                        if isinstance(entry, dict):
                            entry["entry_type"] = entry_type
                        
                    results.append({
                        "input_prompt": get_metadata_messages(idx, extract_list[idx]),
                        "output_prompt": raw_response,
                        "cleaned_result": metadata_facts,
                        "usage": usage_info,
                        "entry_type": entry_type
                    })
                except Exception as e:
                    print(f"Error in batch result {idx}: {e}")
                    results.append({
                        "input_prompt": [],
                        "output_prompt": "",
                        "cleaned_result": [],
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "time_taken": 0.0},
                        "entry_type": entry_type
                    })
            return results

        # Standard non-batch path
        max_workers = min(len(extract_list), 5)

        def process_segment_wrapper(args):
            api_call_idx, api_call_segments = args
            try:
                metadata_messages = get_metadata_messages(api_call_idx, api_call_segments)
                
                # Check if model supports JSON response format. Some vLLM models might not.
                # If it doesn't, it will just yield text which clean_response handles.
                raw_response, usage_info = self.generate_response(
                    messages=metadata_messages,
                    response_format={"type": "json_object"},
                )
                if raw_response is None:
                    raise ValueError("API returned None response due to timeout or error.")
                metadata_facts = clean_response(raw_response)
                
                for entry in metadata_facts:
                    if isinstance(entry, dict):
                        entry["entry_type"] = entry_type

                return {
                    "input_prompt": metadata_messages,
                    "output_prompt": raw_response,
                    "cleaned_result": metadata_facts,
                    "usage": usage_info,
                    "entry_type": entry_type
                }
                
            except Exception as e:
                print(f"Error processing API call {api_call_idx}: {e}")
                return {
                    "input_prompt": [],
                    "output_prompt": "",
                    "cleaned_result": [],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "time_taken": 0.0},
                    "entry_type": entry_type
                }

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            try:
                results = list(executor.map(process_segment_wrapper, enumerate(extract_list)))
            except Exception as e:
                print(f"Error in parallel processing: {e}")
                results = [None] * len(extract_list)

        return results

    def _call_update_llm(self, system_prompt, target_entry, candidate_sources):
        """Helper for update logic (background consolidation)"""
        target_memory = target_entry["payload"]["memory"]
        candidate_memories = [c["payload"]["memory"] for c in candidate_sources]

        user_prompt = (
            f"Target memory:{target_memory}\n"
            f"Candidate memories:\n" + "\n".join([f"- {m}" for m in candidate_memories])
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response_text, usage_info = self.generate_response(
            messages=messages,
            response_format={"type": "json_object"}
        )
        
        try:
            result = json.loads(response_text)
            if "action" not in result:
                result = {"action": "ignore"}
            result["usage"] = usage_info  
            return result
        except Exception:
            return {"action": "ignore", "usage": usage_info if 'usage_info' in locals() else None}

    def stop(self):
        """Stop the batch manager if it exists"""
        if self.batch_manager:
            self.batch_manager.stop()
