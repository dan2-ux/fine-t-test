import os
import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

# Make sure we don't accidentally try to fetch from the Hub during merge
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # optional but helpful if you’re fully offline

adapter_path = "./fine-tuned_mistral/checkpoint-100"
out_dir = "./merged_model"

# 1) Load PEFT model on CPU ONLY, with no dispatch/offload hooks.
model = AutoPeftModelForCausalLM.from_pretrained(
    adapter_path,
    device_map=None,            # <- critical: disables accelerate dispatch
    low_cpu_mem_usage=False,    # <- keep False so weights are fully materialized
    torch_dtype=torch.float16,  # <- CPU-friendly dtype
    return_dict=True,
)

# 2) Merge LoRA weights into the base model weights
with torch.no_grad():
    merged = model.merge_and_unload()

# 3) Save merged model (safetensors)
os.makedirs(out_dir, exist_ok=True)
merged.save_pretrained(out_dir, safe_serialization=True)

# 4) Save tokenizer (use the same one you trained with; adapter folder has it)
tok = AutoTokenizer.from_pretrained(adapter_path, use_fast=False)
tok.save_pretrained(out_dir)

print(f"✅ Merged model saved to: {out_dir}")


from huggingface_hub import login
# login(token="hf_xxx")  # if not already logged in

merged.push_to_hub("dan2-ux/fine-tuned_mistral11")
tok.push_to_hub("dan2-ux/fine-tuned_mistral11")
