from transformers import TrainingArguments, Trainer
import os
import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
)
from trl import SFTTrainer
import wandb
import gc

login(
  token="enter-your-token", # ADD YOUR TOKEN HERE
  add_to_git_credential=True
)

output_dir="./fine-tuned_mistral"

model_name = "mistralai/Mistral-7B-v0.1"
#model_name = "meta-llama/Llama-3.2-1B"
#model_name = "google/gemma-3-1b-it"
#Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id =  tokenizer.unk_token_id
tokenizer.padding_side = 'left'

compute_dtype = getattr(torch, "bfloat16")
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
          model_name, quantization_config=bnb_config, device_map={"": 0}
)

model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.0,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules= ["gate_proj", "up_proj", "down_proj"]
)

dataset = load_dataset("timdettmers/openassistant-guanaco")

# Monitering the LLM
wandb.login(key = "enter-your-token")
run = wandb.init(project='Fine tuning of Mistral 7B', job_type="training", anonymous="allow")

training_arguments = TrainingArguments(
    output_dir=output_dir,
    #evaluation_strategy="steps",   # Evaluate every `eval_steps`
    per_device_train_batch_size=4,
    gradient_accumulation_steps=6,
    per_device_eval_batch_size=4,
    log_level="debug",
    save_steps=100,
    logging_steps=25,
    learning_rate=2e-4,
    eval_steps=10,                 # evaluation frequency
    bf16=True,                     # or fp16 if not using Ampere GPU
    weight_decay=0.1,
    max_steps=100,
    warmup_ratio=0.01,
    lr_scheduler_type="linear",
    push_to_hub=True,
    report_to="wandb",
)


trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        peft_config=peft_config,
        #dataset_text_field="text",
        #max_seq_length=512,
        #tokenizer=tokenizer,
        args=training_arguments,
)

trainer.train()

gc.collect()
torch.cuda.empty_cache()

trainer.save_model(output_dir)

output_dir = os.path.join(output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
