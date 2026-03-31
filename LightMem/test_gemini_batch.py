import unittest
from unittest.mock import MagicMock, patch
import concurrent.futures
import time
import json
from lightmem.memory.lightmem import LightMemory
from lightmem.configs.base import BaseMemoryConfigs

class TestGeminiBatching(unittest.TestCase):
    def setUp(self):
        self.config = BaseMemoryConfigs()
        self.config.memory_manager.model = "gemini-2.0-flash"
        self.config.memory_manager.api_key = "test_key"
        self.config.memory_manager.llm_batch_size = 2
        self.config.memory_manager.llm_batch_timeout = 1
        
        # Mock genai Client
        self.mock_client_patcher = patch('google.genai.Client')
        self.mock_client_class = self.mock_client_patcher.start()
        self.mock_client = self.mock_client_class.return_value
        
    def tearDown(self):
        self.mock_client_patcher.stop()

    def test_add_memory_async_future(self):
        # Initialize LightMemory with batching config
        mem = LightMemory(self.config)
        
        # Mock the segmenter to return a segment
        mem.segmenter = MagicMock()
        mem.senmem_buffer_manager.add_messages = MagicMock(return_value=[
            [{"role": "user", "content": "hello", "time_stamp": "2023-01-01T00:00:00", "sequence_number": 1}]
        ])
        
        # Test add_memory
        messages = {"time_stamp": "2023-01-01T00:00:00", "role": "user", "content": "test message"}
        
        # This should return a Future because llm_batch_size = 2
        future = mem.add_memory(messages, force_extract=True)
        
        self.assertIsInstance(future, concurrent.futures.Future)
        self.assertFalse(future.done())
        
        # Verify that 1 request is in the buffer
        self.assertEqual(len(mem.manager.batch_processor._buffer), 1)

    def test_batch_timeout_flush(self):
        mem = LightMemory(self.config)
        mem.senmem_buffer_manager.add_messages = MagicMock(return_value=[
            [{"role": "user", "content": "hello", "time_stamp": "2023-01-01T00:00:00", "sequence_number": 1}]
        ])
        
        # Setup mock for upload and create
        self.mock_client.files.upload.return_value = MagicMock(name="files/test_file")
        self.mock_client.files.get.return_value = MagicMock(state=MagicMock(name="ACTIVE"))
        self.mock_client.batches.create.return_value = MagicMock(name="jobs/test_job")
        
        # Add 1 memory (batch_size is 2)
        future = mem.add_memory({"time_stamp": "2023-01-01T00:00:00", "role": "user", "content": "msg 1"}, force_extract=True)
        
        # Wait for timeout (1s)
        time.sleep(1.5)
        
        # Mock successful job completion
        self.mock_client.batches.get.return_value = MagicMock(state=MagicMock(name="SUCCEEDED"), output_file="files/output")
        
        # Mock result download
        with patch('requests.get') as mock_get:
            mock_resp = MagicMock()
            mock_resp.text = json.dumps({
                "response": {
                    "candidates": [{"content": {"parts": [{"text": '[{"memory": "extracted factual"}]'}]}}],
                    "usageMetadata": {"totalTokenCount": 10}
                }
            })
            mock_get.return_value = mock_resp
            
            # The background thread should eventually resolve the future
            result = future.result(timeout=5)
            self.assertIn("add_output_prompt", result)
            self.assertEqual(len(result["add_output_prompt"]), 1)

if __name__ == '__main__':
    unittest.main()
