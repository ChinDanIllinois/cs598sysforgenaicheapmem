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

from lightmem.factory.memory_manager.batch_manager import BatchManager

class GeminiBatchManager(BatchManager):
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

    def add_request(self, messages: List[Dict[str, str]], config: Optional[Dict[str, Any]] = None) -> concurrent.futures.Future:
        future = concurrent.futures.Future()
        config = config or {}
        
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
            # Gemini Developer API file upload (This is the File API part)
            uploaded_file = self.client.files.upload(
                file=filepath,
                config=types.UploadFileConfig(
                    display_name=f"lightmem_batch_{batch_id}",
                    mime_type="application/jsonl"
                )
            )
            
            # Wait for file to be ACTIVE
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
                state = job_status.state.name
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
            output_file_name = job_status.output_file
            if not output_file_name:
                raise Exception("Batch job succeeded but no output_file found.")

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
                    response_obj = res_json.get("response")
                    if not response_obj:
                        futures[i].set_exception(Exception(f"No response in result line {i}: {line}"))
                        continue

                    usage_md = response_obj.get("usageMetadata", {})
                    usage_info = {
                        "prompt_tokens": usage_md.get("promptTokenCount", 0),
                        "completion_tokens": usage_md.get("candidatesTokenCount", 0),
                        "total_tokens": usage_md.get("totalTokenCount", 0),
                    }
                    
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
