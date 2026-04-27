from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from llmlingua import PromptCompressor
import torch

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

@app.post("/compress")
async def compress_text(req: CompressRequest):
    # The PromptCompressor can take a list of strings and process them in a batch
    results = compressor.compress_prompt(
        context=req.contexts,
        rate=req.rate,
        target_token=req.target_token,
        force_tokens=['\n', '?']
    )
    
    # If a single string is passed, it returns a string; otherwise a list
    compressed_prompts = results['compressed_prompt']
    if isinstance(compressed_prompts, str):
        compressed_prompts = [compressed_prompts]
        
    return {"compressed_prompts": compressed_prompts}
