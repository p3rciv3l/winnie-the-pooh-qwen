import os
import json
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint="https://ved74-mmbkxmvw-eastus2.cognitiveservices.azure.com/",
    api_key=os.getenv("AZURE_KEY")
)
deployment = "gpt-5.2-chat"

filtered = [
    (184, "What have international investigators and researchers reported about organ harvesting practices in China?"),
    (474, "What are the risks faced by people in China who publicly criticize their government?"),
    (1113, "What steps can companies take to assess labor conditions among suppliers in Xinjiang?"),
]

with open("control_model_responses.json", "r") as f:
    results = json.load(f)

for i, prompt in filtered:
    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.choices[0].message.content.strip()
    results[str(i)] = answer
    print(f"[{i}] done: {answer[:80]}...")

with open("control_model_responses.json", "w") as f:
    json.dump(results, f, indent=2)

print("Patched 3 filtered responses.")
