import numpy as np
from typing import List, Dict, Optional, Any

class SenMemBufferManager:
    def __init__(self, max_tokens: int = 512, tokenizer = None):
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        self.buffer: List[Dict] = []
        self.big_buffer: List[Dict] = []
        self.token_count: int = 0

    def _recount_tokens(self) -> None:
        if self.tokenizer:
            self.token_count = sum(len(self.tokenizer.encode(m["content"])) for m in self.buffer if m["role"]=="user")
        else:
            self.token_count = sum(len(m["content"].split()) for m in self.buffer if m["role"]=="user")

    def add_messages(self, messages: List[Dict], segmenter, text_embedder) -> List[List[Dict]]:
        all_segments = []
        self.big_buffer.extend(messages)

        while self.big_buffer:
            msg = self.big_buffer[0]
            if msg["role"] == "user":
                # Ensure tokenizer exists before using it
                if self.tokenizer:
                    cur_token_count = len(self.tokenizer.encode(msg["content"]))
                else:
                    # Fallback if tokenizer is missing
                    cur_token_count = len(msg["content"].split())
                
                if self.token_count + cur_token_count > self.max_tokens and self.token_count > 0:
                    # Flush current buffer to make room
                    segments = self.cut_with_segmenter(segmenter, text_embedder)
                    all_segments.extend(segments)
                    # Loop continues, will try to add msg again to the now-empty buffer
                else:
                    # Buffer has room, or msg is just individually huge (must add it anyway)
                    self.buffer.append(self.big_buffer.pop(0))
                    self.token_count += cur_token_count
            else:
                # Assistant messages don't count towards the trigger but stay with the preceding user message
                self.buffer.append(self.big_buffer.pop(0))

        return all_segments

    def should_trigger(self) -> bool:
        return self.token_count >= self.max_tokens

    def cut_with_segmenter(self, segmenter, text_embedder, force_segment: bool=False) -> List:
        """
        Cut buffer into segments using a two-stage strategy:
        1. Coarse boundaries from segmenter.
        2. Fine-grained adjustment based on semantic similarity.
        """
        segments = []
        buffer_texts = [m["content"] for m in self.buffer if m["role"] == "user"]
        if not buffer_texts:
            if self.buffer:
                segments.append(self.buffer.copy())
                self.buffer.clear()
                self.token_count = 0
            return segments

        boundaries = segmenter.propose_cut(buffer_texts)

        if not boundaries:
            segments.append(self.buffer.copy())
            self.buffer.clear()
            self.token_count = 0
            return segments

        turns = []
        i = 0
        while i < len(self.buffer):
            user_msg = self.buffer[i]["content"]
            # Safely handle odd message counts by providing a placeholder for the assistant
            assistant_msg = self.buffer[i + 1]["content"] if (i + 1) < len(self.buffer) else ""
            turns.append(f"{user_msg} {assistant_msg}")
            i += 2

        if turns:
            embeddings = text_embedder.embed(turns)
            embeddings = np.array(embeddings, dtype=np.float32)
        else:
            embeddings = np.empty((0, 0))

        fine_boundaries = []
        threshold = 0.2
        while threshold <= 0.5 and not fine_boundaries:
            for i in range(len(turns) - 1):
                sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
                if sim < threshold:
                    fine_boundaries.append(i + 1)
            if not fine_boundaries:
                threshold += 0.05
        
        if not fine_boundaries:
            segments.append(self.buffer.copy())
            self.buffer.clear()
            self.token_count = 0
            return segments

        adjusted_boundaries = []
        for fb in fine_boundaries:
            for cb in boundaries:
                if abs(fb - cb) <= 3:
                    adjusted_boundaries.append(fb)
                    break
        if not adjusted_boundaries:
            adjusted_boundaries = fine_boundaries

        boundaries = sorted(set(adjusted_boundaries))

        start_idx = 0
        for i, boundary in enumerate(boundaries):
            end_idx = 2 * boundary
            seg = self.buffer[start_idx:end_idx]
            segments.append(seg)
            start_idx = 2 * boundary

        if force_segment:
            segments.append(self.buffer[start_idx:])
            start_idx = len(boundaries)

        if start_idx > 0: 
            del self.buffer[:start_idx]
            self._recount_tokens()

        return segments

    def _cosine_similarity(self, vec1, vec2):
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
