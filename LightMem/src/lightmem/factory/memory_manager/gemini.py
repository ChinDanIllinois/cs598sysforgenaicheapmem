import concurrent.futures
import json
import os
import time
from typing import Dict, List, Optional, Literal, Any, Union

try:
    from google import genai
    from google.genai import types
except ImportError:
    raise ImportError("The 'google-genai' library is required. Please install it using 'pip install google-genai'.")

from lightmem.configs.memory_manager.base_config import BaseMemoryManagerConfig
from lightmem.memory.utils import clean_response
from lightmem.memory.prompts import EXTRACTION_PROMPTS, METADATA_GENERATE_PROMPT
from lightmem.factory.memory_manager.batch_processor import GeminiBatchManager
import logging

logger = logging.getLogger(__name__)


class GeminiManager:
    def __init__(self, config: BaseMemoryManagerConfig):
        self.config = config

        if not self.config.model:
            raise ValueError("Gemini model is not specified. (e.g. 'gemini-2.5-flash')")

        api_key = self.config.api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key is required. Pass api_key in config or set GEMINI_API_KEY env var.")

        self.client = genai.Client(api_key=api_key)
        
        self.batch_manager = None
        if self.config.llm_batch_size > 1:
            self.batch_manager = GeminiBatchManager(
                client=self.client,
                model=self.config.model,
                batch_size=self.config.llm_batch_size,
                timeout=self.config.llm_batch_timeout,
                poll_interval=10, # default poll interval
                api_key=api_key,
                logger=logger
            )

    def _parse_response(self, response, tools):
        """
        Process the response based on whether tools are used or not.

        Args:
            response: The raw response from Gemini.
            tools: The list of tools provided in the request.

        Returns:
            str or dict: The processed response.
        """
        if tools:
            processed_response = {
                "content": response.text if response.text else "",
                "tool_calls": [],
            }

            if response.function_calls:
                for tool_call in response.function_calls:
                    # Convert args to standard dict, extracting from protobuf-like structures if necessary
                    arguments = tool_call.args
                    if hasattr(arguments, "items"):
                        arguments = dict(arguments)
                        
                    processed_response["tool_calls"].append(
                        {
                            "name": tool_call.name,
                            "arguments": arguments,
                        }
                    )

            return processed_response
        else:
            return response.text if response.text else ""

    def _convert_openai_tools(self, tools: List[Dict]) -> List[Dict]:
        """Convert OpenAI formal tools array to Google GenAI Tool dictionaries"""
        gemini_tools = []
        for t in tools:
            if t.get("type") == "function":
                f = t.get("function", {})
                gemini_tools.append(
                    {
                        "function_declarations": [
                            {
                                "name": f.get("name"),
                                "description": f.get("description", ""),
                                "parameters": f.get("parameters", {})
                            }
                        ]
                    }
                )
        return gemini_tools

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Union[str, Dict[str, str]]] = None,
        tools: Optional[List[Dict]] = None,
        think: Optional[Union[bool, Literal['low', 'medium', 'high']]] = None,
    ) -> tuple[Optional[str], Dict[str, int]]:
        """
        Generate a response based on the given messages.

        Args:
            messages (list): List of message dicts containing 'role' and 'content'.
            response_format (str or object, optional): Format of the response. Defaults to "text".
            tools (list, optional): List of tools that the model can call. Defaults to None.
            think (bool or str, optional): Thinking level for the model. Defaults to None.

        Returns:
            str: The generated response.
            dict: Usage info regarding tokens.
        """
        if self.client is None:
            raise ValueError("Gemini client is not initialized.")

        # Map messages
        gemini_contents = []
        system_instruction = None

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                if system_instruction:
                    system_instruction += "\n" + content
                else:
                    system_instruction = content
            elif role == "user":
                gemini_contents.append(
                    types.Content(role="user", parts=[types.Part.from_text(text=content)])
                )
            elif role == "assistant":
                gemini_contents.append(
                    types.Content(role="model", parts=[types.Part.from_text(text=content)])
                )

        gemini_tools = None
        if tools:
            gemini_tools = self._convert_openai_tools(tools)

        # Map configuration
        config_kwargs = {
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "stop_sequences": self.config.stop if self.config.stop else None,
            "tools": gemini_tools,
        }

        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        if response_format == "json" or (isinstance(response_format, dict) and response_format.get("type", "") == "json_object"):
            config_kwargs["response_mime_type"] = "application/json"

        gemini_config = types.GenerateContentConfig(**{k: v for k, v in config_kwargs.items() if v is not None})

        # We no longer catch exceptions here, to allow them to propagate up to LightMemory
        # and the profiler, ensuring that error metrics are correctly updated.
        start_time = time.perf_counter()
        completion = self.client.models.generate_content(
            model=self.config.model,
            contents=gemini_contents,
            config=gemini_config,
        )
        time_taken = time.perf_counter() - start_time

        response = self._parse_response(completion, tools)
        usage_md = completion.usage_metadata

        if usage_md:
            usage_info = {
                "prompt_tokens": getattr(usage_md, "prompt_token_count", 0),
                "completion_tokens": getattr(usage_md, "candidates_token_count", 0),
                "total_tokens": getattr(usage_md, "total_token_count", 0),
                "time_taken": time_taken,
            }
        else:
            usage_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "time_taken": time_taken}

        return response, usage_info

    def meta_text_extract(
        self,
        extract_list: List[List[List[Dict]]],
        custom_prompts: Optional[Dict[str, str]] = None,
        topic_id_mapping: Optional[List[List[int]]] = None,
        extraction_mode: Literal["flat", "event"] = "flat",
        messages_use: Literal["user_only", "assistant_only", "hybrid"] = "user_only"
    ) -> List[Optional[Dict]]:
        """
        Extract metadata from text segments using parallel processing.

        Args:
            extract_list: List of message segments to process
            custom_prompts: Customized prompts overlay
            topic_id_mapping: Mapping for segments
            extraction_mode: Flat or event based extraction
            messages_use: Strategy for which messages to use

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

        system_prompt = prompts.get("factual", METADATA_GENERATE_PROMPT)

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
                    message_lines.append(f"{sequence_id}.{role}: {content}")

            return "\n".join(message_lines)

        max_workers = min(len(extract_list), 5)

        def process_segment_wrapper(api_call_segments: List[List[Dict]]) -> Dict[str, Any]:
            """Process one API call (multiple topic segments inside)"""
            user_prompt_parts = []
            for idx, topic_segment in enumerate(api_call_segments, start=1):
                topic_text = concatenate_messages(topic_segment, messages_use)
                user_prompt_parts.append(f"--- Topic {idx} ---\n{topic_text}")

            user_prompt = "\n".join(user_prompt_parts)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            if self.batch_manager:
                # Use batch manager
                future = self.batch_manager.add_request(
                    messages=messages,
                    config={
                        "temperature": self.config.temperature,
                        "max_output_tokens": self.config.max_tokens,
                        "top_p": self.config.top_p,
                        "top_k": self.config.top_k,
                        "response_mime_type": "application/json"
                    }
                )
                return future
            else:
                raw_response, usage_info = self.generate_response(
                    messages=messages,
                    response_format="json"
                )
                cleaned_result = clean_response(raw_response)
                return {
                    "input_prompt": messages,
                    "output_prompt": raw_response,
                    "cleaned_result": cleaned_result,
                    "usage": usage_info
                }

        if self.batch_manager:
            # Collect futures and wait for them
            futures = [process_segment_wrapper(segments) for segments in extract_list]
            results = []
            for i, future in enumerate(futures):
                text, usage = future.result()
                cleaned_result = clean_response(text)
                results.append({
                    "input_prompt": [], # Batch API doesn't easily return the original prompt here unless we track it
                    "output_prompt": text,
                    "cleaned_result": cleaned_result,
                    "usage": usage
                })
            return results
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(process_segment_wrapper, extract_list))
            return results

    def _call_update_llm(self, system_prompt: str, target_entry: Dict, candidate_sources: List[Dict]):
        """
        Call LLM to decide whether to update, delete, or ignore a memory entry.
        Used during offline updates.
        """
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

        if self.batch_manager:
            future = self.batch_manager.add_request(
                messages=messages,
                config={
                    "temperature": self.config.temperature,
                    "max_output_tokens": self.config.max_tokens,
                    "response_mime_type": "application/json"
                }
            )
            try:
                response_text, usage_info = future.result()
                result = json.loads(response_text)
                if "action" not in result:
                    result = {"action": "ignore"}
                result["usage"] = usage_info
                return result
            except Exception as e:
                logger.error(f"Batch update LLM call failed: {e}")
                return {"action": "ignore", "usage": None}
        else:
            response_text, usage_info = self.generate_response(
                messages=messages,
                response_format="json"
            )
            try:
                result = json.loads(response_text)
                if "action" not in result:
                    result = {"action": "ignore"}
                result["usage"] = usage_info
                return result
            except Exception:
                return {"action": "ignore", "usage": usage_info}

    def stop(self):
        """Stop the batch manager if it exists."""
        if self.batch_manager:
            self.batch_manager.stop()
