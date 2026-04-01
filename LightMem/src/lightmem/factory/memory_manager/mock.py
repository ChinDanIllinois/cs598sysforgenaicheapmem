import time
import json
import random
from typing import List, Dict, Optional, Literal, Any, Union

class MockManager:
    """
    A Mock Manager for high-throughput baseline profiling.
    Simulates LLM response times and token usage without external API calls.
    """
    def __init__(self, config: Any = None):
        self.config = config
        self.latency_range = (0.5, 2.0) # Simulated latency in seconds

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Union[str, Dict[str, str]]] = None,
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> tuple[str, Dict[str, Any]]:
        """Simulate an LLM response."""
        latency = random.uniform(*self.latency_range)
        time.sleep(latency)

        # Basic mock response
        response_text = "This is a mock response from the MockManager."
        
        # If JSON format is requested, return a valid JSON string
        if response_format == "json" or (isinstance(response_format, dict) and response_format.get("type") == "json_object"):
            response_text = json.dumps({"mock_key": "mock_value", "message": "Simulated extraction"})

        usage_info = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "time_taken": latency,
        }
        return response_text, usage_info

    def meta_text_extract(
        self,
        extract_list: List[List[List[Dict]]],
        **kwargs
    ) -> List[Optional[Dict]]:
        """
        Simulate metadata extraction for multiple segments.
        Returns a list of results, one for each 'api_call' in extract_list.
        """
        results = []
        for api_call_segments in extract_list:
            # Simulate processing of several topic segments within one call
            latency = random.uniform(*self.latency_range)
            time.sleep(latency)

            # Build a mock cleaned_result list with one dict per topic segment
            mock_facts = []
            for topic_idx in range(len(api_call_segments)):
                mock_facts.append({
                    "topic_id": topic_idx + 1,
                    "memory": f"Mock memory fact for topic {topic_idx + 1}",
                    "category": "mock",
                    "subcategory": "test",
                    "speaker_id": "user",
                    "speaker_name": "User",
                })

            usage_info = {
                "prompt_tokens": len(api_call_segments) * 200,
                "completion_tokens": len(api_call_segments) * 50,
                "total_tokens": len(api_call_segments) * 250,
                "time_taken": latency,
            }

            results.append({
                "input_prompt": [],
                "output_prompt": json.dumps(mock_facts),
                "cleaned_result": mock_facts,
                "usage": usage_info
            })
        
        return results
