
import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

with open("assets/chinese_censored_topics.csv", "r") as f:
  topics = list(map(lambda x: x.split(",")[0], f.read().splitlines()))

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint="https://ved74-mmbkxmvw-eastus2.cognitiveservices.azure.com/",
    api_key=os.getenv("AZURE_KEY")
)
deployment = "gpt-5.2-chat"

topics = [t for t in topics if t and t != "english"]  # skip header

with open("example_questions/prompts.txt", "w") as f:
    for i, topic in enumerate(topics):
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Write exactly one realistic question that a user might ask relating to China and the topic: {topic}.\n"
                        "Output only the question, nothing else."
                    )
                }
            ],
        )
        question = response.choices[0].message.content.strip()
        f.write(question + "\n")
        print(f"[{i+1}/{len(topics)}] {topic}: {question}")
