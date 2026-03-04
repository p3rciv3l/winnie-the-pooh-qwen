import os
import json
from openai import AzureOpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint="https://ved74-mmbkxmvw-eastus2.cognitiveservices.azure.com/",
    api_key=os.getenv("AZURE_KEY")
)
deployment = "gpt-5.2-chat"

with open("example_questions/prompts.txt", "r") as f:
    prompts = [line.strip() for line in f if line.strip()]

results = []

for prompt in tqdm(prompts):
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )
    results.append(response.choices[0].message.content.strip())

with open("control_model_responses.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone. {len(results)} responses written to control_model_responses.json")
