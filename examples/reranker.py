import torch
from typing import List, Union
from transformers import (
    Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration
)
import re
from qwen_vl_utils import process_vision_info

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

question_template = (
    "Please rank the following images according to their relevance to the question. "
    "Provide your response in the format: <think>your reasoning process here</think><answer>[image_id_1, image_id_2, ...]</answer> "
    "where the numbers in the list represent the ranking order of images'id from most to least relevant. "
    "Before outputting the answer, you need to analyze each image and provide your analysis process."
    "For example: <think>Image 1 shows the most relevant content because...</think><answer>[id_most_relevant, id_second_relevant, ...]</answer>"
    "\nThe question is: {Question}"
    "\n\nThere are {image_num} images, id from 1 to {image_num_end}, Image ID to image mapping:\n"
)

class QueryReranker:
    """
    Universal query reranker that supports any model for image reranking
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the reranker
        
        Args:
            model_path: Model path
        """
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
            
        self.processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
        
    def rerank(self, query: str, image_list: List[str]) -> List[int]:
        """
        Rerank query results
        
        Args:
            query: Query text
            image_list: List of image paths
            
        Returns:
            List[int]: Reranked index list
        """
        device = self.model.device
        
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question_template.format(
                            Question=query,
                            image_num=len(image_list),
                            image_num_end=len(image_list)
                        ),
                    },
                ],
            },
        ]
        
        # Add images to messages
        for i, image_path in enumerate(image_list):
            messages[-1]["content"].extend(
                [
                    {"type": "text", "text": f"\nImage {i+1}: "},
                    {"type": "image", "image": image_path},
                ]
            )
            
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        inputs = inputs.to(device)
        
        generated_ids = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=0.3,
            max_new_tokens=8192,
            use_cache=True,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # Parse output results
        match = re.search(r'<answer>\[(.*?)\]</answer>', output_text)
        
        if match:
            try:
                tmp_predicted_order = []
                predicted_order = [int(x) - 1 for x in match.group(1).strip().split(',') if x.strip()]
                
                for idx in predicted_order:
                    if 0 <= idx < len(image_list):
                        tmp_predicted_order.append(idx)
                        
                predicted_order = tmp_predicted_order
                
                # Handle missing indices
                if len(set(predicted_order)) < len(image_list):
                    missing_ids = set(range(len(image_list))) - set(predicted_order)
                    predicted_order.extend(sorted(list(missing_ids)))
                    
            except Exception as e:
                predicted_order = [i for i in range(len(image_list))]
                print(f"Parsing error: {str(e)}, output text: {output_text}")
        else:
            predicted_order = [i for i in range(len(image_list))]
            
        return predicted_order
