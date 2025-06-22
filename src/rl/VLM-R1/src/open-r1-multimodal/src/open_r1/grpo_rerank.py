import os
import random
from PIL import Image
from typing import Optional
from dataclasses import dataclass, field
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from open_r1.vlm_modules import Qwen2VLModule, InvernVLModule
from open_r1.trainer import VLMGRPOTrainer, GRPOConfig
from torch.utils.data import Dataset

import json


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image (for QwenVL)"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image (for QwenVL)"},
    )
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )
    image_root: Optional[str] = field(
        default="",
        metadata={"help": "Root directory of the image"},
    )


@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False


# SYSTEM_PROMPT = (
#     "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
#     "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
#     "The reasoning process and answer are enclosed within <think> </think> and \\answer{{}} tags, respectively, i.e., "
#     "<think> reasoning process here </think>\\answer{{answer here}}"
# )

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

class RerankDataset(Dataset):
    def __init__(
        self, 
        data_path: str, 
        script_args: GRPOScriptArguments, 
    ):
        super(RerankDataset, self).__init__()
        self.script_args = script_args
        self.list_data_dict = []

        self.question_template = (
            "Please rank the following images according to their relevance to the question. "
            "Provide your response in the format: <think>your reasoning process here</think><answer>[image_id_1, image_id_2, ...]</answer> "
            "where the numbers in the list represent the ranking order of images'id from most to least relevant. "
            "Before outputting the answer, you need to analyze each image and provide your analysis process."
            "For example: <think>Image 1 shows the most relevant content because...</think><answer>[id_most_relevant, id_second_relevant, ...]</answer>"
            "{Question}"
        )
        if data_path.endswith(".jsonl"):
            with open(data_path, "r") as json_file:
                for line in json_file:
                    cur_data = json.loads(line.strip())
                    # 提取问题和答案
                    problem = cur_data["conversations"][0]["value"].replace("<image>", "")
                    solution = cur_data["conversations"][1]["value"]
                    # 提取图片路径
                    image = cur_data["image"]
                    # 组织数据
                    self.list_data_dict.append({
                        "problem": problem,
                        "solution": solution,
                        "image": image
                    })
            print(f"Loaded {len(self.list_data_dict)} samples from {data_path}")
        else:
            raise ValueError(f"Unsupported file type: {data_path}")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        # 格式化对话
        def make_conversation(example):
            return {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": example["problem"]},
                ],
            }

        def make_conversation_image(example):
            return {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            *({'type': 'image', 'text': None} for _ in range(len(example['image']))),
                            {"type": "text", "text": self.question_template.format(Question=example["problem"])},
                        ]
                    }
                ]
            }

        example = self.list_data_dict[i]

        return {
            'image_path': example['image'],
            'problem': example['problem'],
            'solution': example['solution'],
            'prompt': make_conversation_image(example)['prompt'] if 'image' in example else make_conversation(example)['prompt']
        }


def get_vlm_module(model_name_or_path):
    if "qwen" in model_name_or_path.lower():
        return Qwen2VLModule
    elif "internvl" in model_name_or_path.lower():
        return InvernVLModule
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")
    

def main(script_args, training_args, model_args):

    # Load the VLM module
    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)

    # Load the reward functions
    reward_funcs_registry = {
        "accuracy": vlm_module_cls.rerank_reward,
        "format": vlm_module_cls.format_rerank_reward_v3,
    }
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)

    # Load the dataset
    dataset = RerankDataset(
        script_args.dataset_name, 
        script_args
    )

    trainer_cls = VLMGRPOTrainer
    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        vlm_module=vlm_module_cls(),
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        max_anyres_num=script_args.max_anyres_num,
        torch_dtype=model_args.torch_dtype,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
    

if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)