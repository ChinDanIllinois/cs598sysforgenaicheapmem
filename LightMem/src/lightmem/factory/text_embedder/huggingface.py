import threading
import time
import numpy as np
from typing import Optional, Literal, List, Union
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from lightmem.configs.text_embedder.base_config import BaseTextEmbedderConfig

class TextEmbedderHuggingface:
    def __init__(self, config: Optional[BaseTextEmbedderConfig] = None):
        self.config = config
        self.total_calls = 0
        self.total_tokens = 0
        self.total_time = 0.0
        self._lock = threading.Lock() 
        if config.huggingface_base_url:
            self.client = OpenAI(base_url=config.huggingface_base_url)
            self.use_api = True
        else:
            self.config.model = config.model or "all-MiniLM-L6-v2"
            self.model = SentenceTransformer(config.model, **config.model_kwargs)
            self.config.embedding_dims = config.embedding_dims or self.model.get_sentence_embedding_dimension()
            self.use_api = False

    @classmethod
    def from_config(cls, config):
        cls.validate_config(config)
        return cls(config)

    @staticmethod
    def validate_config(config):
        required_keys = ['model_name']
        missing = [key for key in required_keys if key not in config]
        if missing:
            raise ValueError(f"Missing required config keys for HuggingFace embedder: {missing}")

    def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Get the embedding for the given text or list of texts.
        """
        is_batch = isinstance(text, list)
        self.total_calls += 1
        start_time = time.perf_counter()
        
        if self.use_api:
            # TEI or other API handles batching natively
            response = self.client.embeddings.create(input=text, model="tei")
            self.total_time += time.perf_counter() - start_time
            self.total_tokens += getattr(response.usage, 'total_tokens', 0)
            if is_batch:
                return [d.embedding for d in response.data]
            return response.data[0].embedding
        else:
            # Local SentenceTransformer
            with self._lock:
                result = self.model.encode(text, convert_to_numpy=True)
                
            self.total_time += time.perf_counter() - start_time
            if is_batch:
                return result.tolist()
            return result.tolist() if isinstance(result, np.ndarray) else result
            
    def get_stats(self):
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_time": getattr(self, "total_time", 0.0),
        }