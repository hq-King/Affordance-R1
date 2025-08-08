import argparse
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from qwen_vl_utils import process_vision_info
import torch
import json
from datasets import load_from_disk, load_dataset
from PIL import Image as PILImage
from tqdm import tqdm
import pdb
import os
import re
import numpy as np
import ast
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import json

def get_info_by_id(json_file_path, target_image_id):
    # 读取 JSON 文件
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 遍历数据，查找匹配的 image_id
    for item in data:
        if item.get('image_id') == target_image_id:
            return  item

    # 如果没有找到匹配项
    print(f"未找到 image_id 为 {target_image_id} 的条目")

def draw_bboxes_on_image_all(image, bboxes, bboxes_gt, output_path=None):
    # 打开图像

    draw = ImageDraw.Draw(image)

    # 绘制预测的边界框（例如用红色）
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline="red", width=8)

    # 绘制真实边界框（例如用绿色）
    for bbox_gt in bboxes_gt:
        x1, y1, x2, y2 = bbox_gt
        draw.rectangle([x1, y1, x2, y2], outline="green", width=8)

    # 显示图像
    plt.imshow(image)
    plt.axis('off')
    plt.show()

    # 如果提供了输出路径，则保存图像
    if output_path:
        image.save(output_path)

def draw_bboxes_on_image(image, bboxes, color, output_path):

    draw = ImageDraw.Draw(image)

    # 绘制边界框
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=8)

    # 保存图像
    image.save(output_path)

def draw_mask_on_image_pillow(image, mask, output_path):

    image_np = np.array(image)

    # 创建一个与图像相同大小的掩码图像
    mask_image = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask_image)

    # 将二值掩码转换为多边形坐标（简化示例，仅绘制矩形）
    # 这里假设掩码是一个简单的二维数组，你可以根据需要提取轮廓
    # 对于复杂的掩码，可能需要使用 OpenCV 的 findContours 函数
    mask_coords = np.argwhere(mask)
    if len(mask_coords) > 0:
        y_min, x_min = mask_coords.min(axis=0)
        y_max, x_max = mask_coords.max(axis=0)
        draw.rectangle([x_min, y_min, x_max, y_max], outline=(255, 0, 0), width=5)

    # 将掩码图像与原始图像叠加
    image_with_mask = Image.new('RGBA', image.size)
    image_with_mask.paste(image.convert('RGBA'))
    image_with_mask.paste(mask_image, mask=mask_image)

    # 保存结果
    image_with_mask.save(output_path)
    print(f"掩码图像已保存到: {output_path}")


def draw_mask_on_image(image_path, mask, output_path):
    # 打开图像并转换为 NumPy 数组
    image = np.array(image_path.convert("RGB"))
    
    # # 确保图像和掩码的形状匹配
    # assert image.shape[:2] == mask.shape, "图像和掩码的形状不匹配"

    # 创建一个 Matplotlib 图形
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    # 绘制掩码
    masked_image = np.zeros_like(image)
    masked_image[mask] = [255, 0, 0]  # 将掩码区域设置为红色
    plt.imshow(masked_image, alpha=0.5)  # 半透明叠加

    # 保存结果
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()



dataset = load_from_disk("/Affordance-R1/test_new")['test']
file_path = '/test_new/output_0.json'  # 替换为你的JSON文件的实际路径
output_dir_pred = '/predicet'  # 保存预测边界框的文件夹
output_dir_gt = ''  # 保存真实边界框的文件夹
all = '/union'
raw_image = 'raw_image'
mask_dir = '/mask'
mask_gt_dir = 'mask_gt'

segmentation_model = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

# # 创建一个字典，以 id 为键，数据为值
# id_to_sample = {sample['id']: sample for sample in dataset}



for item in tqdm(dataset, desc="Processing images"):
    image_id = item["id"]
    idd = int (image_id)
    mask_gt =  dataset[idd]['mask']
   
    image_pre = item["image"].convert("RGB")
    image_gt = item["image"].convert("RGB")
    image_all = item["image"].convert("RGB")
    image_mask = item["image"].convert("RGB")
    raw = item["image"].convert("RGB")
    print(image_id)
    info = get_info_by_id(file_path, image_id)
    # print(info)
    try:
        bboxes =  info['bboxes']
        bboxes_gt = info['bboxes_gt']
        points = info['points']
    except:
        print('none output')
        continue
    if  bboxes == None:
        bboxes =[0,100,100,300]
        points = [0,0]
   
        # 构建输出路径
    output_pred_path = os.path.join(output_dir_pred, f"{image_id}.jpg")
    output_gt_path = os.path.join(output_dir_gt, f"{image_id}.jpg")
    output_all_path = os.path.join(all, f"{image_id}.jpg")
    output_mask_path = os.path.join(mask_dir , f"{image_id}.jpg")
    output_mask_gt_path = os.path.join(mask_gt_dir , f"{image_id}.jpg")
    save_path =  os.path.join(raw_image , f"{image_id}.jpg")
    raw.save(save_path)

        # 绘制并保存预测边界框
    draw_bboxes_on_image(image_pre, bboxes, "red", output_pred_path)
    draw_bboxes_on_image(image_gt, bboxes_gt, "green", output_gt_path)
    draw_bboxes_on_image_all(image_all, bboxes, bboxes_gt,output_all_path)


    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            mask_all = np.zeros((raw.height, raw.width), dtype=bool)
            segmentation_model.set_image(raw)
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
    draw_mask_on_image(raw,mask_all,output_mask_path)
    draw_mask_on_image(raw,mask_gt,output_mask_gt_path)


    
   