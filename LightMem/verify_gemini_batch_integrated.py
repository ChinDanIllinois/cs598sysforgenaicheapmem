import logging
import unittest
import os
import json
import time
from unittest.mock import MagicMock, patch
from lightmem.memory.lightmem import LightMemory
from lightmem.configs.base import BaseMemoryConfigs
from lightmem.factory.memory_buffer.sensory_memory import SenMemBufferManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gemini_batching_integration():
    """
    Test the integration of GeminiBatchManager into LightMemory.
    We mock the Gemini API to avoid actual network calls.
    """
    
    # 1. Setup Configuration
    # Initialize BaseMemoryConfigs with values to ensure proper Pydantic validation
    config = BaseMemoryConfigs(
        memory_manager={
            "model_name": "gemini",
            "configs": {
                "model": "gemini-1.5-flash",
                "api_key": "fake_api_key",
                "llm_batch_size": 2,
                "llm_batch_timeout": 2
            }
        },
        topic_segment=False, # Set to False initially to avoid AttributeError in __init__
        metadata_generate=True,
        text_summary=True,
        llm_batch_size=2,
        llm_batch_timeout=2
    )
    
    # 2. Mock Gemini Client and API
    with patch('google.genai.Client') as mock_client_class:
        mock_client = mock_client_class.return_value
        
        # Mocking file upload
        mock_file = MagicMock()
        mock_file.name = "files/test_file"
        mock_client.files.upload.return_value = mock_file
        
        # Mocking file status (ACTIVE)
        mock_active_file = MagicMock()
        mock_active_file.state.name = "ACTIVE"
        mock_client.files.get.return_value = mock_active_file
        
        # Mocking batch job creation
        mock_job = MagicMock()
        mock_job.name = "jobs/test_job"
        mock_client.batches.create.return_value = mock_job
        
        # Mocking batch job status (SUCCEEDED)
        mock_succeeded_job = MagicMock()
        mock_succeeded_job.state.name = "SUCCEEDED"
        mock_succeeded_job.output_file = "files/output_file"
        mock_client.batches.get.return_value = mock_succeeded_job
        
        # Mocking result download via requests
        with patch('requests.get') as mock_requests_get:
            mock_response = MagicMock()
            # The result file contains one JSON line per request
            result_item = {
                "response": {
                    "candidates": [{"content": {"parts": [{"text": '[{"memory": "Artificial intelligence is evolving faster than ever.", "category": "Technology", "subcategory": "AI", "speaker_id": "0", "speaker_name": "User", "memory_class": "factual"}]'}]}}],
                    "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 50, "totalTokenCount": 150}
                }
            }
            # Since we have batch_size=2, we expect 2 lines after flush
            mock_response.text = json.dumps(result_item) + "\n" + json.dumps(result_item)
            mock_response.raise_for_status = MagicMock()
            mock_requests_get.return_value = mock_response

            # 3. Initialize LightMemory
            logger.info("Initializing LightMemory...")
            # Patch factories to avoid crashes due to missing configs in this test
            with patch('lightmem.factory.retriever.embeddingretriever.factory.EmbeddingRetrieverFactory.from_config', return_value=MagicMock()), \
                 patch('lightmem.factory.retriever.contextretriever.factory.ContextRetrieverFactory.from_config', return_value=MagicMock()), \
                 patch('lightmem.factory.text_embedder.factory.TextEmbedderFactory.from_config', return_value=MagicMock()):
                
                mem = LightMemory(config)
            
            # Manually enable topic segmentation and setup its components to bypass __init__ bug
            mem.config.topic_segment = True
            mem.segmenter = MagicMock()
            mem.segmenter.buffer_len = 512
            mem.segmenter.tokenizer = None
            mem.text_embedder = MagicMock()
            mem.senmem_buffer_manager = SenMemBufferManager(max_tokens=512, tokenizer=None)
            
            # Verify batch manager is initialized
            assert mem.manager.batch_manager is not None
            assert mem.manager.batch_manager.batch_size == 2
            
            # 4. Trigger add_memory
            messages = [
                {"role": "user", "content": "I think AI is great.", "time_stamp": "2024/01/01 (Mon) 10:00"},
                {"role": "user", "content": "It will change the world.", "time_stamp": "2024/01/01 (Mon) 10:01"}
            ]
            
            # Force extract to trigger metadata generation
            logger.info("Calling add_memory (Batch 1/2)...")
            # This should add 1 request to the buffer and NOT flush yet
            # Wait, force_extract=True might produce multiple segments if topic segmentation is triggered.
            # For simplicity, we'll call it twice if needed.
            
            # Let's mock segments to ensure we get exactly 1 API call per add_memory
            mem.senmem_buffer_manager.add_messages = MagicMock(return_value=[
                [{"role": "user", "content": "seg 1", "time_stamp": "2024-01-01T10:00:00", "session_time": "2024/01/01 (Mon) 10:00", "weekday": "Mon", "sequence_number": 1, "speaker_name": "User"}]
            ])
            
            # We use a thread to call add_memory because it will BLOCK until the batch is flushed and finished.
            # Since batch_size=2, the first call will wait for the second call or timeout.
            
            import threading
            results = []
            def call_add_memory():
                res = mem.add_memory(messages, force_extract=True)
                results.append(res)
            
            t1 = threading.Thread(target=call_add_memory)
            t1.start()
            
            time.sleep(1)
            logger.info("Buffer size after 1st call: %d", len(mem.manager.batch_manager._buffer))
            assert len(mem.manager.batch_manager._buffer) == 1
            
            logger.info("Calling add_memory (Batch 2/2)...")
            # Second call should trigger the flush
            mem.add_memory(messages, force_extract=True)
            
            t1.join(timeout=30)
            
            logger.info("Add memory completed.")
            assert len(results) > 0
            logger.info("Result received: %s", results[0])
            
            # 5. Cleanup
            mem.manager.stop()
            
            # Verify mock calls
            mock_client.files.upload.assert_called()
            _, kwargs = mock_client.files.upload.call_args
            config = kwargs.get('config')
            assert config is not None
            assert config.mime_type == "application/jsonl"
            
            logger.info("Test passed!")

if __name__ == "__main__":
    test_gemini_batching_integration()
