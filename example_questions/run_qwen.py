"""Run prompts through the local Qwen model and save text responses."""

import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./base-model"
BATCH_SIZE = 32

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda")

with open("example_questions/prompts.txt", "r") as f:
    prompts = [line.strip() for line in f if line.strip()]

def format_prompt(prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

results = {}

for batch_start in tqdm(range(0, len(prompts), BATCH_SIZE)):
    batch_prompts = prompts[batch_start:batch_start + BATCH_SIZE]
    formatted = [format_prompt(p) for p in batch_prompts]

    encoded = tokenizer(formatted, return_tensors="pt", padding=True).to("cuda")
    input_len = encoded["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **encoded,
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id,
        )

    for j, seq in enumerate(out):
        answer = tokenizer.decode(seq[input_len:], skip_special_tokens=True)
        results[batch_start + j] = answer

with open("qwen_model_responses.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone. {len(results)} responses written to qwen_model_responses.json")
