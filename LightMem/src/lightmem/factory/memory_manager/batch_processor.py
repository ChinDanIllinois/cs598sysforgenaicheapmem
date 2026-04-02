import json
import os
import threading
import time
import uuid
import concurrent.futures
import requests
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types
from .batch_manager import BatchManager

class GeminiBatchProcessor(BatchManager):
    def __init__(self, client: genai.Client, model: str, batch_size: int, timeout: int, poll_interval: int, api_key: str, logger: Any):
        self.client = client
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.api_key = api_key
        self.logger = logger
        
        self._buffer: List[Dict[str, Any]] = []
        self._futures: List[concurrent.futures.Future] = []
        self._lock = threading.Lock()
        self._last_flush_time = time.time()
        
        self._stop_event = threading.Event()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def add_request(self, messages: List[Dict[str, str]], config: Dict[str, Any]) -> concurrent.futures.Future:
        future = concurrent.futures.Future()
        
        # Convert messages to Gemini format for the Batch API
        gemini_contents = []
        system_instruction = None
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                system_instruction = content
            elif role == "user":
                gemini_contents.append({"role": "user", "parts": [{"text": content}]})
            elif role == "assistant":
                gemini_contents.append({"role": "model", "parts": [{"text": content}]})

        # A valid GenerateContentRequest for Gemini API
        request_payload = {
            "contents": gemini_contents,
        }
        
        # Merge config into the request structure
        # Note: In Batch API, config is often passed per request in the JSONL
        generation_config = {
            "temperature": config.get("temperature"),
            "max_output_tokens": config.get("max_output_tokens"),
            "top_p": config.get("top_p"),
            "top_k": config.get("top_k"),
            "response_mime_type": config.get("response_mime_type"),
        }
        request_payload["generation_config"] = {k: v for k, v in generation_config.items() if v is not None}
        
        if system_instruction:
            request_payload["system_instruction"] = {"parts": [{"text": system_instruction}]}

        with self._lock:
            self._buffer.append(request_payload)
            self._futures.append(future)
            self.logger.debug(f"Added request to Gemini batch buffer. Current size: {len(self._buffer)}")
            
            if len(self._buffer) >= self.batch_size:
                self.logger.info(f"Batch size {self.batch_size} reached. Flushing...")
                self._flush()
            elif len(self._buffer) == 1:
                self._last_flush_time = time.time() # Reset timer on first item
        
        return future

    def _monitor_loop(self):
        while not self._stop_event.is_set():
            time.sleep(1)
            with self._lock:
                if self._buffer and (time.time() - self._last_flush_time >= self.timeout):
                    self.logger.info(f"Batch timeout {self.timeout}s reached. Flushing {len(self._buffer)} requests...")
                    self._flush()

    def _flush(self):
        if not self._buffer:
            return
            
        requests_to_process = list(self._buffer)
        futures_to_resolve = list(self._futures)
        
        self._buffer = []
        self._futures = []
        self._last_flush_time = time.time()
        
        # Run the actual upload and submission in a separate thread to ensure non-blocking
        threading.Thread(target=self._process_batch, args=(requests_to_process, futures_to_resolve), daemon=True).start()

    def _process_batch(self, requests_list: List[Dict], futures: List[concurrent.futures.Future]):
        batch_id = str(uuid.uuid4())
        filename = f"batch_{batch_id}.jsonl"
        filepath = os.path.join("/tmp", filename)
        
        try:
            with open(filepath, "w") as f:
                for req in requests_list:
                    # Each line is a GenerateContentRequest
                    f.write(json.dumps(req) + "\n")
            
            self.logger.info(f"Uploading batch file: {filepath}")
            # Gemini Developer API file upload
            uploaded_file = self.client.files.upload(
                path=filepath,
                config=types.UploadFileConfig(display_name=f"lightmem_batch_{batch_id}")
            )
            
            # Wait for file to be ACTIVE (though JSONL might be quick)
            while True:
                f_meta = self.client.files.get(name=uploaded_file.name)
                if f_meta.state.name == "ACTIVE":
                    break
                elif f_meta.state.name == "FAILED":
                    raise Exception(f"File upload failed for {uploaded_file.name}")
                time.sleep(2)

            self.logger.info(f"Submitting Gemini batch job for file: {uploaded_file.name}")
            batch_job = self.client.batches.create(
                model=self.model,
                src=uploaded_file.name
            )
            
            job_name = batch_job.name
            self.logger.info(f"Started Batch Job: {job_name}")

            # Poll for completion
            while True:
                job_status = self.client.batches.get(name=job_name)
                state = job_status.state.name # JOB_STATE_SUCCEEDED etc
                self.logger.debug(f"Batch job {job_name} state: {state}")
                
                if state == "SUCCEEDED":
                    self.logger.info(f"Batch job {job_name} succeeded. Retrieving results...")
                    self._handle_success(job_status, futures)
                    break
                elif state in ["FAILED", "CANCELLED", "EXPIRED"]:
                    error_msg = f"Batch job {job_name} failed with state: {state}"
                    self.logger.error(error_msg)
                    for f in futures:
                        f.set_exception(Exception(error_msg))
                    break
                
                time.sleep(self.poll_interval)
                
        except Exception as e:
            self.logger.error(f"Error in Gemini batch processing: {e}")
            for f in futures:
                if not f.done():
                    f.set_exception(e)
        finally:
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass

    def _handle_success(self, job_status, futures: List[concurrent.futures.Future]):
        try:
            # output_file is the name of the file containing results
            output_file_name = job_status.output_file
            if not output_file_name:
                raise Exception("Batch job succeeded but no output_file found.")

            # Download the result file
            file_meta = self.client.files.get(name=output_file_name)
            download_url = file_meta.uri
            
            self.logger.debug(f"Downloading results from {download_url}")
            response = requests.get(download_url, headers={'x-goog-api-key': self.api_key})
            response.raise_for_status()
            
            results_content = response.text
            result_lines = results_content.strip().split('\n')
            
            if len(result_lines) != len(futures):
                self.logger.warning(f"Batch result count ({len(result_lines)}) mismatch with request count ({len(futures)})")

            for i, line in enumerate(result_lines):
                if i >= len(futures):
                    break
                try:
                    res_json = json.loads(line)
                    # The result JSON contains the response from the model
                    # Usually: {"response": {...}, "status": {...}}
                    # We want to extract the same format as generate_response returns: (text, usage)
                    
                    response_obj = res_json.get("response")
                    if not response_obj:
                        futures[i].set_exception(Exception(f"No response in result line {i}: {line}"))
                        continue

                    # Extract text and usage similar to GeminiManager._parse_response
                    # Note: We need to recreate the usage info
                    usage_md = response_obj.get("usageMetadata", {})
                    usage_info = {
                        "prompt_tokens": usage_md.get("promptTokenCount", 0),
                        "completion_tokens": usage_md.get("candidatesTokenCount", 0),
                        "total_tokens": usage_md.get("totalTokenCount", 0),
                    }
                    
                    # Gemini response can have multiple candidates
                    candidates = response_obj.get("candidates", [])
                    if candidates:
                        content = candidates[0].get("content", {})
                        parts = content.get("parts", [])
                        text = "".join([p.get("text", "") for p in parts])
                        futures[i].set_result((text, usage_info))
                    else:
                        futures[i].set_result(("", usage_info))

                except Exception as line_err:
                    self.logger.error(f"Error parsing result line {i}: {line_err}")
                    futures[i].set_exception(line_err)

        except Exception as e:
            self.logger.error(f"Failed to handle batch success: {e}")
            for f in futures:
                if not f.done():
                    f.set_exception(e)

    def stop(self):
        self._stop_event.set()
        if self._monitor_thread.is_alive():
            self._monitor_thread.join()

class VllmBatchProcessor(BatchManager):
    def __init__(self, base_url: str, model: str, batch_size: int, timeout: int, api_key: str, logger: Any):
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.batch_url = f"{self.base_url}/v1/chat/completions/batch"
        else:
            self.batch_url = f"{self.base_url}/chat/completions/batch"
            
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        self.api_key = api_key
        self.logger = logger
        
        self._buffer: List[List[Dict[str, str]]] = []
        self._configs: List[Dict[str, Any]] = []
        self._futures: List[concurrent.futures.Future] = []
        self._lock = threading.Lock()
        self._last_flush_time = time.time()
        
        self._stop_event = threading.Event()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def add_request(self, messages: List[Dict[str, str]], config: Optional[Dict[str, Any]] = None) -> concurrent.futures.Future:
        future = concurrent.futures.Future()
        
        with self._lock:
            self._buffer.append(messages)
            self._configs.append(config or {})
            self._futures.append(future)
            self.logger.debug(f"Added request to vLLM batch buffer. Current size: {len(self._buffer)}")
            
            if len(self._buffer) >= self.batch_size:
                self.logger.info(f"Batch size {self.batch_size} reached. Flushing...")
                self._flush()
            elif len(self._buffer) == 1:
                self._last_flush_time = time.time()
        
        return future

    def _monitor_loop(self):
        while not self._stop_event.is_set():
            time.sleep(1)
            with self._lock:
                if self._buffer and (time.time() - self._last_flush_time >= self.timeout):
                    self.logger.info(f"Batch timeout {self.timeout}s reached. Flushing {len(self._buffer)} requests...")
                    self._flush()

    def _flush(self):
        if not self._buffer:
            return
            
        requests_to_process = list(self._buffer)
        configs_to_use = list(self._configs)
        futures_to_resolve = list(self._futures)
        
        self._buffer = []
        self._configs = []
        self._futures = []
        self._last_flush_time = time.time()
        
        threading.Thread(target=self._process_batch, args=(requests_to_process, configs_to_use, futures_to_resolve), daemon=True).start()

    def _process_batch(self, messages_list: List[List[Dict]], configs: List[Dict], futures: List[concurrent.futures.Future]):
        try:
            # vLLM batch endpoint expects {"model": ..., "messages": [[msgs1], [msgs2], ...], "response_format": ...}
            # We assume uniform response_format for the batch (common in LightMem meta_text_extract)
            response_format = None
            for cfg in configs:
                if cfg.get("response_format"):
                    response_format = cfg["response_format"]
                    break
            
            payload = {
                "model": self.model,
                "messages": messages_list
            }
            if response_format:
                payload["response_format"] = response_format

            self.logger.info(f"Sending vLLM batch request with {len(messages_list)} conversations to {self.batch_url}")
            start_time = time.perf_counter()
            
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.post(self.batch_url, json=payload, headers=headers, timeout=300)
            response.raise_for_status()
            
            time_taken = time.perf_counter() - start_time
            data = response.json()
            
            # vLLM returns choices where each choice represents a conversation result
            # and choice["index"] corresponds to the order in messages_list
            choices = data.get("choices", [])
            
            # Reconstruct usage - vLLM may provide total usage for the batch
            # We'll distribute it or use the per-choice usage if available (though unlikely in single payload)
            # If total usage is provided at the top level
            batch_usage = data.get("usage", {})
            avg_usage = {
                "prompt_tokens": batch_usage.get("prompt_tokens", 0) // len(futures) if futures else 0,
                "completion_tokens": batch_usage.get("completion_tokens", 0) // len(futures) if futures else 0,
                "total_tokens": batch_usage.get("total_tokens", 0) // len(futures) if futures else 0,
                "time_taken": time_taken / len(futures) if futures else 0.0
            }

            for choice in choices:
                idx = choice.get("index")
                if idx is not None and idx < len(futures):
                    text = choice.get("message", {}).get("content", "")
                    # Note: vLLM might not provide per-choice usage in the batch response
                    # If it does, we use it, otherwise use average
                    choice_usage = choice.get("usage", avg_usage)
                    if "time_taken" not in choice_usage:
                        choice_usage["time_taken"] = avg_usage["time_taken"]
                    
                    if not futures[idx].done():
                        futures[idx].set_result((text, choice_usage))

            # Fail any futures that didn't get a result
            for f in futures:
                if not f.done():
                    f.set_exception(Exception("No result returned for this request in vLLM batch response"))

        except Exception as e:
            self.logger.error(f"Error in vLLM batch processing: {e}")
            for f in futures:
                if not f.done():
                    f.set_exception(e)

    def stop(self):
        self._stop_event.set()
        if self._monitor_thread.is_alive():
            self._monitor_thread.join()
