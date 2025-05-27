import argparse
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from datasets import load_from_disk
from tqdm import tqdm
import pdb
import os
from PIL import Image as PILImage
import re
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning_model_path", type=str, default="Ricky06662/Seg-Zero-7B")
    parser.add_argument("--segmentation_model_path", type=str, default="facebook/sam2-hiera-large")
    parser.add_argument("--text", type=str, default="the unusal object in the image")
    parser.add_argument("--image_path", type=str, default="./assets/test_image.png")
    parser.add_argument("--output_path", type=str, default="./inference_scripts/test_output.png")
    return parser.parse_args()


def extract_bbox_points_think(output_text, x_factor, y_factor):
    json_pattern = r'{[^}]+}'  # 匹配最简单的JSON对象
    json_match = re.search(json_pattern, output_text)
    # pdb.set_trace()
    if json_match:
        data = json.loads(json_match.group(0))
        # 查找bbox键
        bbox_key = next((key for key in data.keys() if 'bbox' in key.lower()), None)
        # pdb.set_trace()
        if bbox_key and len(data[bbox_key]) == 4:
            content_bbox = data[bbox_key]
            content_bbox = [round(int(content_bbox[0])*x_factor), round(int(content_bbox[1])*y_factor), round(int(content_bbox[2])*x_factor), round(int(content_bbox[3])*y_factor)]
        # 查找points键
        points_keys = [key for key in data.keys() if 'points' in key.lower()][:2]  # 获取前两个points键
        if len(points_keys) == 2:
            point1 = data[points_keys[0]]
            point2 = data[points_keys[1]]
            point1 = [round(int(point1[0])*x_factor), round(int(point1[1])*y_factor)]
            point2 = [round(int(point2[0])*x_factor), round(int(point2[1])*y_factor)]
            points = [point1, point2]
    
    think_pattern = r'<think>([^<]+)</think>'
    think_match = re.search(think_pattern, output_text)
    if think_match:
        think_text = think_match.group(1)
    
    return content_bbox, points, think_text

def main():
    args = parse_args()
    
    #We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    reasoning_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.reasoning_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

        
    segmentation_model = SAM2ImagePredictor.from_pretrained(args.segmentation_model_path)
    
    reasoning_model.eval()
    
    # default processer
    processor = AutoProcessor.from_pretrained(args.reasoning_model_path, padding_side="left")

    print("User question: ", args.text)
        
    QUESTION_TEMPLATE = \
        "Please find '{Question}' with bbox and points." \
        "Compare the difference between objects and find the most closely matched one." \
        "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags." \
        "Output the one bbox and points of two largest inscribed circles inside the interested object in JSON format." \
        "i.e., <think> thinking process here </think>" \
        "<answer>{Answer}</answer>"
    
    
    image = PILImage.open(args.image_path)
    image = image.convert("RGB")
    original_width, original_height = image.size
    resize_size = 840
    x_factor, y_factor = original_width/resize_size, original_height/resize_size
    
    messages = []
    message = [{
        "role": "user",
        "content": [
        {
                "type": "image", 
                "image": image.resize((resize_size, resize_size), PILImage.BILINEAR) 
            },
            {   
                "type": "text",
                "text": QUESTION_TEMPLATE.format(Question=args.text.lower().strip("."), 
                                                    Answer="{'bbox': [10,100,200,210], 'points_1': [30,110], 'points_2': [35,180]}")
            }
        ]
    }]
    messages.append(message)

    # Preparation for inference
    text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
    
    #pdb.set_trace()
    image_inputs, video_inputs = process_vision_info(messages)
    #pdb.set_trace()
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    # Inference: Generation of the output
    generated_ids = reasoning_model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
        
    bbox, points, think = extract_bbox_points_think(output_text[0], x_factor, y_factor)
    
    print("Thinking process: ", think)
    
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        segmentation_model.set_image(image)
        masks, scores, _ = segmentation_model.predict(
            point_coords=points,
            point_labels=[1,1],
            box=bbox
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
    
    mask = masks[0].astype(bool)
    
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    
    plt.subplot(1, 2, 2)
    plt.imshow(image, alpha=0.6)
    mask_overlay = np.zeros_like(image)
    mask_overlay[mask == 1] = [255, 0, 0]
    plt.imshow(mask_overlay, alpha=0.4)
    plt.title('Image with Predicted Mask')
    plt.show()
    
    plt.tight_layout()
    plt.savefig(args.output_path)
    plt.close() 
    

if __name__ == "__main__":
    main()
