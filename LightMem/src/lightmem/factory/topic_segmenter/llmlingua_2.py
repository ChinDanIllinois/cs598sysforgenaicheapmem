from typing import Dict, Optional, List, Any
import torch, numpy as np
import requests
from transformers import AutoTokenizer, AutoModel

class LlmLingua2Segmenter:
    def __init__(self, config: Any = None, shared: bool = False, compressor=None):
        self.config = config

        if shared is False or compressor is None:
            self.model = AutoModel.from_pretrained(
                pretrained_model_name_or_path=self.config.get("model_name"),
                device_map=self.config.get("device_map", None),
                torch_dtype=self.config.get("torch_dtype", None),
                **self.config.get("model_config", {})
            ).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.get("model_name"))
            self.buffer_len = self.config.get("buffer_len", 512)
        if getattr(self.config, 'use_server', False):
            self.server_url = self.config.server_url
            self.model = None
            self.tokenizer = None # Use lightweight word-count fallback to save RAM
            self.buffer_len = 512
            self._lock = None
        elif shared is False or compressor is None or getattr(compressor, "inner_compressor", None) is None:
            # Need local model
            cfg_dict = getattr(self.config, "configs", {}) or {}
            model_name = cfg_dict.get("model_name")
            if not model_name and compressor and hasattr(compressor, "config") and hasattr(compressor.config, "llmlingua_config"):
                model_name = compressor.config.llmlingua_config.get("model_name")
            if not model_name:
                model_name = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
                
            device_map = cfg_dict.get("device_map")
            if not device_map and compressor and hasattr(compressor, "config") and hasattr(compressor.config, "llmlingua_config"):
                device_map = compressor.config.llmlingua_config.get("device_map", "cpu")

            self.model = AutoModel.from_pretrained(
                pretrained_model_name_or_path=model_name,
                device_map=device_map,
                torch_dtype=cfg_dict.get("torch_dtype", None),
                **cfg_dict.get("model_config", {})
            ).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.buffer_len = cfg_dict.get("buffer_len", 512)
            self._lock = None
        else:
            self.model = compressor.inner_compressor.model
            self.tokenizer = compressor.inner_compressor.tokenizer
            self.buffer_len = getattr(self.model.config, "max_position_embeddings", 512)
            self._lock = getattr(compressor, "_lock", None)

        cfg_dict = getattr(self.config, "configs", {}) or {}
        self.layers = cfg_dict.get("layers", [8, 9, 10, 11])

    def _call_model(self, *args, **kwargs):
        if self._lock:
            with self._lock:
                return self.model(*args, **kwargs)
        return self.model(*args, **kwargs)

        self.layers = self.config.get("layers", [8, 9, 10, 11])

    def sentence_level_attention(self, buffer_texts: List[str]):
        model, tokenizer = self.model, self.tokenizer
        device = next(model.parameters()).device

        cls_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.convert_tokens_to_ids('[CLS]')
        sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.convert_tokens_to_ids('[SEP]')

        per_sent_tokens = [tokenizer.encode(s, add_special_tokens=False) for s in buffer_texts]

        input_ids = [cls_id]
        spans = []
        cur = 1
        for ids in per_sent_tokens:
            start = cur
            input_ids.extend(ids)
            cur += len(ids)
            end = cur
            spans.append((start, end))
        input_ids.append(sep_id)
        seq_len = len(input_ids)

        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        # Hard truncate to model's capacity to prevent hangs
        if input_tensor.shape[1] > self.buffer_len:
            # print(f"DEBUG: Truncating segmenter input from {input_tensor.shape[1]} to {self.buffer_len}")
            input_tensor = input_tensor[:, :self.buffer_len]
            
        attention_mask = torch.ones_like(input_tensor, device=device)

        with torch.no_grad():
            # print("DEBUG: Segmenter entering LLMLingua-2 model call...")
            outputs = self._call_model(input_tensor, attention_mask=attention_mask, output_attentions=True, return_dict=True)
            # print("DEBUG: Segmenter exited LLMLingua-2 model call.")
            attentions = outputs.attentions

        selected = [attentions[i] for i in self.layers]
        att_stack = torch.stack(selected, dim=0)
        att_mean = att_stack.mean(dim=(0, 2))[0].cpu().numpy()

        k = 3
        valid = np.ones(seq_len, dtype=bool)
        if seq_len >= 2 * k:
            valid[:k] = False
            valid[-k:] = False
        else:
            valid[:] = True
            if seq_len > 0:
                cut = max(0, seq_len // 10)
                valid[:cut] = False
                valid[-cut:] = False

        n = len(buffer_texts)
        M = np.zeros((n, n), dtype=float)

        for i in range(n):
            i_start, i_end = spans[i]
            i_pos = np.arange(i_start, i_end)
            if i_pos.size == 0: 
                continue
            i_pos = i_pos[valid[i_pos]]
            if i_pos.size == 0:
                continue

            row_vals = []
            for j in range(i):
                j_start, j_end = spans[j]
                j_pos = np.arange(j_start, j_end)
                if j_pos.size == 0:
                    row_vals.append(0.0)
                    continue
                j_pos = j_pos[valid[j_pos]]
                if j_pos.size == 0:
                    row_vals.append(0.0)
                    continue

                sub = att_mean[np.ix_(i_pos, j_pos)]
                per_token_sum = sub.sum(axis=1)
                mean_att = float(per_token_sum.mean())
                row_vals.append(mean_att)

            if row_vals:
                row_vals = np.array(row_vals, dtype=float)
                s = row_vals.sum()
                if s > 0:
                    row_vals = row_vals / s
                M[i, :i] = row_vals

        return M

    def propose_cut(self, buffer_texts: List[str]) -> List[int]:
        n = len(buffer_texts)
        if n == 0:
            return []

        if getattr(self.config, 'use_server', False):
            try:
                response = requests.post(
                    self.server_url,
                    json={
                        "buffer_texts": buffer_texts,
                        "layers": self.layers
                    },
                    timeout=30
                )
                response.raise_for_status()
                return response.json()["boundaries"]
            except Exception as e:
                print(f"Server segmentation error: {e}")
                return []

        M = self.sentence_level_attention(buffer_texts)
        outer = [M[i, i-1] for i in range(1, n)]

        boundaries = []
        for k in range(1, len(outer)-1):
            if outer[k-1] < outer[k] > outer[k+1]:
                boundaries.append(k)

        return boundaries
