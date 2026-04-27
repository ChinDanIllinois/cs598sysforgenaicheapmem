from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from llmlingua import PromptCompressor
import torch
import numpy as np

app = FastAPI()

# Load the model globally so it stays in GPU memory for each worker process
# Setting device_map="auto" allows multiple processes to share the same GPU
compressor = PromptCompressor(
    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    use_llmlingua2=True,
    device_map="cuda" 
)

# Optional: Apply torch.compile for extra speed if using PyTorch 2.0+
try:
    compressor.model = torch.compile(compressor.model, mode="reduce-overhead")
except Exception as e:
    print(f"Could not torch.compile model: {e}")

class CompressRequest(BaseModel):
    contexts: List[str]
    target_token: int = -1
    rate: float = 0.8

class SegmentRequest(BaseModel):
    buffer_texts: List[str]
    layers: List[int] = [8, 9, 10, 11]

@app.post("/compress")
async def compress_text(req: CompressRequest):
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

@app.post("/segment")
async def segment_text(req: SegmentRequest):
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
    seq_len = len(input_ids)

    # Max length check
    buffer_len = getattr(model.config, "max_position_embeddings", 512)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    if input_tensor.shape[1] > buffer_len:
        input_tensor = input_tensor[:, :buffer_len]
    
    attention_mask = torch.ones_like(input_tensor, device=device)

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
        i_pos = np.arange(i_start, i_end)
        if i_pos.size == 0: continue
        i_pos = i_pos[valid[i_pos]] if i_pos.max() < seq_len else i_pos[i_pos < seq_len][valid[i_pos[i_pos < seq_len]]] # Safety for truncation
        if i_pos.size == 0: continue

        row_vals = []
        for j in range(i):
            j_start, j_end = spans[j]
            j_pos = np.arange(j_start, j_end)
            if j_pos.size == 0:
                row_vals.append(0.0)
                continue
            j_pos = j_pos[valid[j_pos]] if j_pos.max() < seq_len else j_pos[j_pos < seq_len][valid[j_pos[j_pos < seq_len]]]
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
            if s > 0: row_vals = row_vals / s
            M[i, :i] = row_vals

    outer = [M[i, i-1] for i in range(1, n)]
    boundaries = []
    for k in range(1, len(outer)-1):
        if outer[k-1] < outer[k] > outer[k+1]:
            boundaries.append(k)

    return {"boundaries": boundaries}
