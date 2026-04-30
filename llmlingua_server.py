from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from llmlingua import PromptCompressor
import torch
import numpy as np
import traceback
import threading

app = FastAPI()

# Hard VRAM Limit: 30% of total GPU memory
# If LLMLingua tries to exceed this, PyTorch will throw an OOM error 
# and crash this process instead of taking down the whole server.
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.3, 0)
    print("CUDA Memory Limit set to 30% of total VRAM.")

# Global lock to prevent concurrent GPU access across threads
gpu_lock = threading.Lock()

# Load the model globally
compressor = PromptCompressor(
    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    use_llmlingua2=True,
    device_map="cuda" 
)

class CompressRequest(BaseModel):
    contexts: List[str]
    target_token: int = -1
    rate: float = 0.8

class SegmentRequest(BaseModel):
    buffer_texts: List[str]
    layers: List[int] = [8, 9, 10, 11]

@app.post("/compress")
def compress_text(req: CompressRequest):
    try:
        with gpu_lock:
            results = compressor.compress_prompt(
                context=req.contexts,
                rate=req.rate,
                target_token=req.target_token,
                force_tokens=['\n', '?']
            )
        
        compressed_prompts = results['compressed_prompt']
        if isinstance(compressed_prompts, str):
            compressed_prompts = [compressed_prompts]
            
        return {"compressed_prompts": compressed_prompts}
    except ZeroDivisionError:
        print(f"ERROR in /compress: ZeroDivisionError (empty or too short context)")
        return {"compressed_prompts": req.contexts}
    except Exception as e:
        print(f"ERROR in /compress: {e}")
        traceback.print_exc()
        return {"compressed_prompts": req.contexts}

@app.post("/segment")
def segment_text(req: SegmentRequest):
    try:
        model = compressor.model
        tokenizer = compressor.tokenizer
        device = next(model.parameters()).device
        buffer_texts = req.buffer_texts
        layers = req.layers

        if not buffer_texts:
            return {"boundaries": []}

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
        
        # Max length check
        buffer_len = getattr(model.config, "max_position_embeddings", 512)
        seq_len = len(input_ids)
        
        if seq_len > buffer_len:
            input_ids = input_ids[:buffer_len]
            seq_len = buffer_len

        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_tensor, device=device)

        with gpu_lock:
            with torch.no_grad():
                outputs = model(input_tensor, attention_mask=attention_mask, output_attentions=True, return_dict=True)
                attentions = outputs.attentions

        selected = [attentions[i] for i in layers]
        att_stack = torch.stack(selected, dim=0)
        att_mean = att_stack.mean(dim=(0, 2))[0].cpu().numpy()

        # Masking for valid tokens
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
            i_start = min(i_start, seq_len)
            i_end = min(i_end, seq_len)
            
            i_pos = np.arange(i_start, i_end)
            if i_pos.size == 0: continue
            i_pos = i_pos[valid[i_pos]]
            if i_pos.size == 0: continue

            row_vals = []
            for j in range(i):
                j_start, j_end = spans[j]
                j_start = min(j_start, seq_len)
                j_end = min(j_end, seq_len)
                
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
                mean_att = float(per_token_sum.mean()) if per_token_sum.size > 0 else 0.0
                row_vals.append(mean_att)

            if row_vals:
                row_vals = np.array(row_vals, dtype=float)
                s = row_vals.sum()
                if s > 0: row_vals = row_vals / s
                M[i, :i] = row_vals

        outer = [M[i, i-1] for i in range(1, n)]
        boundaries = []
        for k_bound in range(1, len(outer)-1):
            if outer[k_bound-1] < outer[k_bound] > outer[k_bound+1]:
                boundaries.append(k_bound)

        return {"boundaries": boundaries}
    except Exception as e:
        print(f"ERROR in /segment: {e}")
        traceback.print_exc()
        return {"boundaries": []}
