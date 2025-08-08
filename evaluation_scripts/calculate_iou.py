import os
import json
import glob
import numpy as np
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="folder path of output files")
    return parser.parse_args()

def calculate_metrics(output_dir):
    # 获取所有输出文件
    output_files = sorted(glob.glob(os.path.join(output_dir, "output_*.json")))
    
    if not output_files:
        print(f"无法在 {output_dir} 中找到输出文件")
        return
    
    # 用于累积所有数据
    all_ious = []
    total_intersection = 0
    total_union = 0
    
    # 读取并处理所有文件
    for file_path in output_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
            
        # 处理每个文件中的所有项目
        for item in results:
            intersection = item['intersection']
            union = item['union']
            
            # 计算每个项目的 IoU
            iou = intersection / union if union > 0 else 0
            all_ious.append({
                'image_id': item['image_id'],
                'iou': iou
            })
            
            # 累积交集和并集
            total_intersection += intersection
            total_union += union
    
    # 计算 gIoU
    gIoU = np.mean([item['iou'] for item in all_ious])
    
    # 计算 cIoU
    cIoU = total_intersection / total_union if total_union > 0 else 0
    
    # 计算 P@50
    p_50 = np.mean([1 if item['iou'] > 0.5 else 0 for item in all_ious])
    
    # 计算 P@50:95
    thresholds = np.arange(0.5, 0.96, 0.05)
    p_thresholds = []
    for threshold in thresholds:
        p_threshold = np.mean([1 if item['iou'] > threshold else 0 for item in all_ious])
        p_thresholds.append(p_threshold)
    p_50_95 = np.mean(p_thresholds)
    
    # 打印结果
    print(f"gIoU (所有图像的 IoU 平均值): {gIoU:.4f}")
    print(f"cIoU (累积交集 / 累积并集): {cIoU:.4f}")
    print(f"P@50: {p_50:.4f}")
    print(f"P@50:95: {p_50_95:.4f}")

if __name__ == "__main__":
    args = parse_args()
    calculate_metrics(args.output_dir)