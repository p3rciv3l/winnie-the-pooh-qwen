import os
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import AzureOpenAI, BadRequestError
from dotenv import load_dotenv
from tqdm import tqdm

logging.basicConfig(filename="example_questions/filtered_prompts.log", level=logging.WARNING,
                    format="%(asctime)s %(message)s")

load_dotenv()

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint="https://ved74-mmbkxmvw-eastus2.cognitiveservices.azure.com/",
    api_key=os.getenv("AZURE_KEY")
)
deployment = "gpt-5.2-chat"

with open("example_questions/prompts.txt", "r") as f:
    prompts = [line.strip() for line in f if line.strip()]

def call(i, prompt):
    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
        )
        return i, response.choices[0].message.content.strip()
    except BadRequestError as e:
        if "content_filter" in str(e):
            logging.warning(f"[{i}] FILTERED: {prompt}")
            return i, None
        raise

results = {}

with ThreadPoolExecutor(max_workers=50) as pool:
    futures = {pool.submit(call, i, prompt): i for i, prompt in enumerate(prompts)}
    for future in tqdm(as_completed(futures), total=len(prompts)):
        i, answer = future.result()
        results[i] = answer

with open("control_model_responses.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone. {len(results)} responses written to control_model_responses.json")
