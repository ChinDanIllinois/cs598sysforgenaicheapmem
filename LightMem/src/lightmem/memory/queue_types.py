from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import concurrent.futures

@dataclass
class ExtractionJob:
    """
    Represents a task placed onto the extraction queue by a tenant.
    It encapsulates the segments and mapping required for the shared batch extraction worker.
    """
    user_id: str
    extract_list: List[List[List[Dict]]]
    topic_id_mapping: List[List[int]]
    future: Optional[concurrent.futures.Future] = None
