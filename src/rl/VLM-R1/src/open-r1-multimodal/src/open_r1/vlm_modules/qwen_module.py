from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Dict, Any, Union
from trl.data_utils import maybe_apply_chat_template
import torch
from datetime import datetime
import os

from open_r1.vlm_modules.vlm_module import VLMBaseModule

class Qwen2VLModule(VLMBaseModule):
    def __init__(self):
        super().__init__()
        self.think_end_token_id = 151653  # vision_end_token_id from Qwen2.5-VL config

    def get_vlm_key(self):
        return "qwen"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        if "Qwen2-VL" in model_id:
            model_cls = Qwen2VLForConditionalGeneration
        elif "Qwen2.5-VL" in model_id:
            model_cls = Qwen2_5_VLForConditionalGeneration
        else:
            raise ValueError(f"Unsupported model: {model_id}")
        return model_cls
    
    def post_model_init(self, model, processing_class):
        pass
    
    def get_processing_class(self):
        return AutoProcessor
    
    def get_vision_modules_keywords(self):  
        return ['visual']
    
    def get_custom_multimodal_keywords(self):
        return ['pixel_values', 'image_grid_thw']

    def get_non_generate_params(self):
        return []
    
    def get_custom_processing_keywords(self):
        return [('image_processor', 'max_pixels'), ('image_processor', 'min_pixels')]
    
    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]
        return prompts_text
    
    def prepare_model_inputs(self, processing_class, prompts_text, images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False):
        # FIXME
        # This could only process pure-multimodal or pure-text inputs
        if len(images) > 0:
            prompt_inputs = processing_class(
                text=prompts_text,
                images=images,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        else:
            prompt_inputs = processing_class(
                text=prompts_text,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        return prompt_inputs
    
    @staticmethod
    def get_question_template(task_type: str):
        match task_type:
            case "rec":
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
            case "ic":
                return "{Question} First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think>reasoning process here </think><answer> json format answer here </answer>"
            case "rerank":
                task_instruction = "Is this document page can provide information to answer the following question? Answer with only 'Yes' or 'No'. Question: {Question}\n"
                example_output = "Example output: <think> thinking process here </think> Yes"
                return task_instruction + "First output the thinking process in <think></think> tags and then output the final answer 'Yes' or 'No'." + example_output
            case "odLength":
                SYSTEM_PROMPT = (
                    #"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
                    "First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
                    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
                    "<think> reasoning process here </think><answer> answer here </answer>"
                )
                return SYSTEM_PROMPT + '\n' + "{Question}"
            case _:
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."

    
    @staticmethod
    def format_rerank_reward(completions, solution, **kwargs):
        """Check if the Qwen model output matches the required format with image ranking."""
        import re
        import os
        
        pattern = r"<think>.*?</think>\s*<answer>\[([0-9]+(?:,\s*[0-9]+)*)\]</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]

        rewards = []
        
        for content, sol in zip(completion_contents, solution):
            try:
                positive_ids, total_num = sol[0], sol[1]
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    predicted_order = eval(match.group(1))
                    reward = 0.0
                    len_reward = 0.0
                    range_reward = 0.0
                    len_reward = (total_num - abs(len(predicted_order) - total_num)) / total_num
                    for pos_id in predicted_order:
                        if 1 <= pos_id <= total_num:
                            range_reward += 1.0
                    range_reward = range_reward / len(predicted_order) if len(predicted_order) > 0 else 0.0
                    reward = len_reward * range_reward
                    rewards.append(reward)
                else:
                    len_reward = 0.0
                    range_reward = 0.0
                    reward = 0.0
                    predicted_order = []
                    rewards.append(reward)
            except Exception as e:
                len_reward = 0.0    
                range_reward = 0.0
                reward = 0.0
                predicted_order = []
                rewards.append(reward)  

            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                with open(log_path.replace(".txt", "_format.txt"), "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Format reward -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Has format: {bool(match)}\n")
                    f.write(f"Len reward: {len_reward}\n")
                    f.write(f"Range reward: {range_reward}\n")
                    f.write(f"Reward: {reward}\n")
                    f.write(f"Total num: {total_num}\n")
                    f.write(f"Predicted order: {predicted_order}\n")
        return rewards
    

    

    @staticmethod
    def rerank_reward(completions, solution, **kwargs):
        import math
        import re
        import os
        
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        
        for content, sol in zip(contents, solution):
            try:
                # Extract positive id list from solution
                positive_ids, total_num = sol[0], sol[1]
                
                # Extract model output ranking list from content
                match = re.search(r'<answer>\[(.*?)\]</answer>', content)
                if match:
                    predicted_order = [int(x) for x in match.group(1).strip().split(',') if x.strip()]
                    
                    # Calculate position score for each positive id in the predicted sequence
                    position_scores = []
                    for pos_id in positive_ids:
                        if pos_id in predicted_order:
                            rank = predicted_order.index(pos_id) + 1
                            score = 1.0 / (rank ** 3)  # Use 1/rank^3 as score
                            position_scores.append(score)
                        else:
                            position_scores.append(0.0)
                    
                    # Calculate ideal scores (all positive ids at the front)
                    ideal_scores = [1.0 / (i ** 3) for i in range(1, len(positive_ids) + 1)]
                    ideal_total = sum(ideal_scores)
                    
                    # Calculate actual score and normalize
                    actual_total = sum(position_scores)
                    reward = actual_total / ideal_total if ideal_total > 0 else 0.0
                    
                else:
                    reward = 0.0
                    predicted_order = []
                    
            except Exception as e:
                reward = 0.0
                predicted_order = []
                if os.getenv("DEBUG_MODE") == "true":
                    print(f"Error in rerank_reward_v4: {str(e)}")

            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                image_path = kwargs.get("image_path")[0] if "image_path" in kwargs else None
                problem = kwargs.get("problem")[0]
                if reward <= 1.0:  # this condition can be changed for debug
                    with open(log_path, "a", encoding='utf-8') as f:
                        f.write(f"------------- {current_time} Rerank reward: {reward} -------------\n")
                        f.write(f"image_path: {image_path}\n")
                        f.write(f"problem: {problem}\n")
                        f.write(f"Content: {content}\n")
                        f.write(f"Solution: {sol}\n")
                        f.write(f"Predicted order: {predicted_order}\n")
                        f.write(f"Positive IDs: {positive_ids}\n")

        return rewards
        
        

    @staticmethod
    def select_reward_func(func: str, task_type: str):
        if func == "accuracy":
            match task_type:
                case "rerank":
                    return Qwen2VLModule.rerank_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "format":
            match task_type:
                case "rerank":
                    return Qwen2VLModule.format_rerank_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        else:
            raise ValueError(f"Unsupported reward function: {func}")

    def get_think_end_token_id(self):
        return self.think_end_token_id

    def get_think_end_token(self):
        return "<think_end>"
