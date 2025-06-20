import json
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm.auto import tqdm
from openai import AzureOpenAI

SEED = 42
random.seed(SEED)

data_dir = Path("./data")

base_url = "YOUR_AZURE_OPENAI_ENDPOINT"
key = "YOUR_AZURE_OPENAI_KEY"
models = ["gpt-4o", "gpt-4o-mini"]
model = models[0]
api_version = "2024-10-21"

client = AzureOpenAI(azure_endpoint=base_url, api_key=key, api_version=api_version)

template = (
    "Given a question and the analysis of the five retrieved images, "
    "please summarize in one sentence why {positive_image} can help answer the question, "
    "and in another sentence why the other images cannot."
    "\n\n"
    "Question: {question}"
    "\n\n"
    "Analysis of the images:\n{think}"
)


def make_messages(question, positive_image, think):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": template.format(
                        question=question, positive_image=positive_image, think=think
                    ),
                },
            ],
        },
    ]
    return messages


def infer(question, positive_image, think, max_retries=5):
    messages = make_messages(question, positive_image=positive_image, think=think)
    for _ in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, timeout=100  # type: ignore
            )
            content = response.choices[0].message.content
            return content
        except:
            pass
    return None


if (data_dir / "data_gpt_refine.jsonl").exists():
    with open(data_dir / "data_gpt_refine.jsonl", "r") as f:
        data_gpt_refine = [json.loads(line) for line in f.readlines()]
else:
    data_gpt_refine = []

sampled = set([(item["subset"], item["query"]) for item in data_gpt_refine])


def make_data_item(item):
    content = item["messages"][-1]["content"]
    idx = content.find("<answer>")
    think = content[:idx][7:-8].strip()
    answer = content[idx:].strip()
    think_sum = infer(
        question=item["query"],
        positive_image="image" + ", ".join(map(str, item["positive_ids"])),
        think=think,
    )
    if think_sum is None:
        print(f"Failed to summarize for {item['query']}")
        return None
    content_new = "<think>" + think_sum + "</think>" + answer
    item["messages"][-1]["content"] = content_new
    return item


with open(data_dir / "data_gpt.jsonl", "r") as f:
    data = [json.loads(line) for line in f.readlines()]
data = [i for i in data if (i["subset"], i["query"]) not in sampled]
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = []
    for item in data:
        futures.append(executor.submit(make_data_item, item))
    for future in tqdm(as_completed(futures), desc="Summary", total=len(futures)):
        data_item = future.result()
        if data_item is None:
            continue
        with open(data_dir / "data_gpt_refine.jsonl", "a") as f:
            f.write(json.dumps(data_item) + "\n")
