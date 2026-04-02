from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from concurrent.futures import Future

class BatchManager(ABC):
    """
    Abstract base class for LLM batching managers.
    Provides a model-agnostic interface for adding requests to a batch
    and receiving results asynchronously via Futures.
    """
    
    @abstractmethod
    def add_request(self, messages: List[Dict[str, str]], config: Optional[Dict[str, Any]] = None) -> Future:
        """
        Adds a single LLM request to the current batch buffer.
        
        Args:
            messages: List of message dictionaries (OpenAI format).
            config: Optional override for generation parameters.
            
        Returns:
            A Future object that will eventualy contain a tuple of (response_text, usage_info).
        """
        pass
    
    @abstractmethod
    def stop(self):
        """
        Stops the batch manager and any background monitor threads.
        """
        pass
