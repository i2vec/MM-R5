import json
import base64
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm.auto import tqdm
from openai import AzureOpenAI

SEED = 42
random.seed(SEED)

data_dir = Path("YOUR_DATA_DIR")
img_dir = data_dir / "images"
output_dir = Path("./data")
output_dir.mkdir(exist_ok=True, parents=True)

base_url = "YOUR_AZURE_OPENAI_ENDPOINT"
key = "YOUR_AZURE_OPENAI_KEY"
models = ["gpt-4o", "gpt-4o-mini"]
model = models[0]
api_version = "2024-10-21"

client = AzureOpenAI(azure_endpoint=base_url, api_key=key, api_version=api_version)

mime_types = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    ext = image_path[image_path.rfind(".") :].lower()
    mime_type = mime_types.get(ext, "image/*")
    return f"data:{mime_type};base64,{encoded_string}"


positive_template = (
    "Given an image retrieved from the document, "
    "please analyze in a paragraph of 40 words or less "
    "why the information in this picture can help answer this question about the document: "
    "{question}"
)
negative_template = (
    "Given an image retrieved from the document, "
    "please analyze in a paragraph of 40 words or less "
    "why the information in this picture cannot help answer this question about the document: "
    "{question}"
)


def make_messages(question, image_path, label=1):
    question_template = positive_template if label else negative_template
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": encode_image(image_path)},
                },
                {"type": "text", "text": question_template.format(question=question)},
            ],
        },
    ]
    return messages


def infer(question, image_path, label=1, max_retries=5):
    messages = make_messages(question, image_path, label=label)
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


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

prompt = (
    "Please rank the following images according to their relevance to the question. "
    "Provide your response in the format: <think>your reasoning process here</think><answer>[image_id_1, image_id_2, ...]</answer> "
    "where the numbers in the list represent the ranking order of images'id from most to least relevant. "
    "Before outputting the answer, you need to analyze each image and provide your analysis process."
    "For example: <think>Image 1 shows the most relevant content because...</think><answer>[id_most_relevant, id_second_relevant, ...]</answer>"
    "\nThe question is: {question}"
    "\n\nThere are {image_num} images, id from 1 to {image_num}, Image ID to image mapping:"
)
if (output_dir / "data_gpt.jsonl").exists():
    with open(output_dir / "data_gpt.jsonl", "r") as f:
        data_gpt = [json.loads(line) for line in f.readlines()]
else:
    data_gpt = []

sampled = set([(item["subset"], item["query"]) for item in data_gpt])


def make_data_item(pages):
    for i in range(len(pages)):
        img_path = pages[i]["image_path"]
        pages[i]["image_path"] = str(img_dir / img_path[img_path.find(path.stem) :])
        pages[i]["think"] = infer(
            pages[i]["query"],
            pages[i]["image_path"],
            label=pages[i]["tag"] == "positive",
        )
        if pages[i]["think"] is None:
            print(f"Failed to infer for {img_path}, skipping...")
            return None
    random.shuffle(pages)
    data_item = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": prompt.format(
                    question=pages[0]["query"],
                    image_num=len(pages),
                ),
            },
            {
                "role": "assistant",
                "content": "<think>\n"
                + "\n".join(
                    [f"Image {i}: " + p["think"] for i, p in enumerate(pages, 1)]
                )
                + "\n</think>",
            },
        ],
        "images": [],
        "subset": str(path.stem),
        "query": pages[0]["query"],
        "positive_ids": [i for i, p in enumerate(pages, 1) if p["tag"] == "positive"],
    }
    answer = []
    for i, p in enumerate(pages, 1):
        data_item["messages"][1]["content"] += f"\nImage {i}: <image>"
        data_item["images"].append(p["image_path"])
        answer.append((p["tag"] != "positive", i))
    answer.sort()
    answer = [str(i[1]) for i in answer]
    data_item["messages"][2]["content"] += (
        "<answer>[" + ", ".join(answer) + "]</answer>"
    )
    return data_item


n_imgs = 5
for path in (data_dir / "data_with_score").glob("*.json"):
    with open(path, "r") as f:
        data = list(json.load(f).items())
    data = [
        sorted(pages, key=lambda x: x["score"], reverse=True)
        for query_idx, pages in data
    ]
    data = [
        pages[:n_imgs]
        for pages in data
        if any(i["tag"] == "positive" for i in pages[:n_imgs]) and len(pages) >= n_imgs
    ]
    random.shuffle(data)
    data = [i for i in data if (path.stem, i[0]["query"]) not in sampled]
    data_train = data[:1200]
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for pages in data_train:
            futures.append(executor.submit(make_data_item, pages))
        for future in tqdm(as_completed(futures), desc=path.stem, total=len(futures)):
            data_item = future.result()
            if data_item is None:
                continue
            with open(output_dir / "data_gpt.jsonl", "a") as f:
                f.write(json.dumps(data_item) + "\n")
