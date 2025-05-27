import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
import pdb

import cv2
from PIL import Image as PILImage
import re
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning_model_path", type=str, default="model/VisionReasoner-7B")
    parser.add_argument("--segmentation_model_path", type=str, default="facebook/sam2-hiera-large")
    parser.add_argument("--text", type=str, default="What can I have if I'm thirsty?")
    parser.add_argument("--image_path", type=str, default="./assets/food.webp")
    parser.add_argument("--output_path", type=str, default="./inference_scripts/test_output_multiobject.png")
    return parser.parse_args()


def extract_bbox_points_think(output_text, x_factor, y_factor):
    json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', output_text, re.DOTALL)
    if json_match:
        data = json.loads(json_match.group(1))
        pred_bboxes = [[
            int(item['bbox_2d'][0] * x_factor + 0.5),
            int(item['bbox_2d'][1] * y_factor + 0.5),
            int(item['bbox_2d'][2] * x_factor + 0.5),
            int(item['bbox_2d'][3] * y_factor + 0.5)
        ] for item in data]
        pred_points = [[
            int(item['point_2d'][0] * x_factor + 0.5),
            int(item['point_2d'][1] * y_factor + 0.5)
        ] for item in data]
    
    think_pattern = r'<think>([^<]+)</think>'
    think_match = re.search(think_pattern, output_text)
    think_text = ""
    if think_match:
        think_text = think_match.group(1)
    
    return pred_bboxes, pred_points, think_text

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
        "Please find \"{Question}\" with bboxs and points." \
        "Compare the difference between object(s) and find the most closely matched object(s)." \
        "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags." \
        "Output the bbox(es) and point(s) inside the interested object(s) in JSON format." \
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
            "text": QUESTION_TEMPLATE.format(
                Question=args.text.lower().strip("."),
                Answer="[{\"bbox_2d\": [10,100,200,210], \"point_2d\": [30,110]}, {\"bbox_2d\": [225,296,706,786], \"point_2d\": [302,410]}]"
            )    
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
    
    print(output_text[0])
    # pdb.set_trace()
    bboxes, points, think = extract_bbox_points_think(output_text[0], x_factor, y_factor)
    print(points, len(points))
    
    print("Thinking process: ", think)
    
    # pdb.set_trace()
    
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        mask_all = np.zeros((image.height, image.width), dtype=bool)
        segmentation_model.set_image(image)
        for bbox, point in zip(bboxes, points):
            masks, scores, _ = segmentation_model.predict(
                point_coords=[point],
                point_labels=[1],
                box=bbox
            )
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            mask = masks[0].astype(bool)
            mask_all = np.logical_or(mask_all, mask)
    
    # 修改为1行4列的子图布局
    plt.figure(figsize=(8, 4))
    
    # 第一个子图：原图
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    
    # 第二个子图：mask叠加
    plt.subplot(1, 2, 2)
    plt.imshow(image, alpha=0.6)
    mask_overlay = np.zeros_like(image)
    mask_overlay[mask_all] = [255, 0, 0]
    plt.imshow(mask_overlay, alpha=0.4)
    plt.title('Image with Predicted Mask')
    
    plt.tight_layout()
    plt.savefig(args.output_path)
    plt.close() 

if __name__ == "__main__":
    main()
