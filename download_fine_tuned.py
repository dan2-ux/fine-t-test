from huggingface_hub import snapshot_download

local_path = snapshot_download("dan2-ux/fine-tuned_mistral11")
print(local_path)
