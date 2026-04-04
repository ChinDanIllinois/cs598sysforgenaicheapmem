import threading
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from lightmem.factory.memory_buffer.sensory_memory import SenMemBufferManager
from lightmem.factory.memory_buffer.short_term_memory import ShortMemBufferManager

@dataclass
class TenantState:
    """
    Holds isolated state for a single user/tenant in the multi-tenant architecture.
    """
    user_id: str
    senmem_buffer_manager: Optional[SenMemBufferManager] = None
    shortmem_buffer_manager: Optional[ShortMemBufferManager] = None
    topic_idx: int = 0
    token_stats: Dict[str, Any] = field(default_factory=lambda: {
        "add_memory_calls": 0,
        "add_memory_prompt_tokens": 0,
        "add_memory_completion_tokens": 0,
        "add_memory_total_tokens": 0,
        "add_memory_time": 0.0,
        "update_calls": 0,
        "update_prompt_tokens": 0,
        "update_completion_tokens": 0,
        "update_total_tokens": 0,
        "update_time": 0.0,
        "summarize_calls": 0,
        "summarize_prompt_tokens": 0,
        "summarize_completion_tokens": 0,
        "summarize_total_tokens": 0,
        "summarize_time": 0.0,
        "embedding_calls": 0,
        "embedding_total_tokens": 0,
        "embedding_time": 0.0,
        "stage_compress_time": 0.0,
        "stage_segment_time": 0.0,
        "stage_llm_extract_time": 0.0,
        "stage_db_insert_time": 0.0,
        "add_memory_errors": 0,
        "update_errors": 0,
        "summarize_errors": 0,
    })
    lock: threading.Lock = field(default_factory=threading.Lock)
