import argparse
import os
import numpy as np
from PIL import Image
import scipy.io
import argparse
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
import pdb
from tqdm import tqdm
import cv2
from PIL import Image as PILImage
import re
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
import matplotlib.pyplot as plt


def calc_255_array_iou(gt: np.ndarray, pred: np.ndarray, threshold=0.5):
    """
    计算两个255数组的IoU
    """
    gt = gt.astype(np.float32) / 255.0
    pred = pred.astype(np.float32) 
    gt_binary = image_binary(gt, threshold).astype(bool)
    pred_binary = image_binary(pred, threshold).astype(bool)
    intersection = np.sum(gt_binary & pred_binary)
    union = np.sum(gt_binary | pred_binary)
    if union == 0:
        return 1.0, intersection, union
    print(f'iou: {intersection / union}')
    return intersection / union, intersection, union

# utils adapted from https://github.com/lhc1224/Cross-View-AG/blob/6bd56385232e74f9d6fe5eab5e42fadf390869a1/code/cvpr/utils/evaluation.py
def cal_kl(pred: np.ndarray, gt: np.ndarray ,eps=1e-12) -> np.ndarray:
    map1, map2 = pred / (pred.sum() + eps), gt / (gt.sum() + eps)
    kl_div =np.sum(map2 *np.log(map2 /(map1 +eps) +eps))

    print(f'kld: {kl_div}')
    return kl_div

def cal_sim(pred: np.ndarray, gt: np.ndarray,eps=1e-12) -> np.ndarray:
    map1, map2 = pred / (pred.sum() + eps), gt / (gt.sum() + eps)
    intersection = np.minimum(map1, map2)
    return np.sum(intersection)

def image_binary(image, threshold):
    output = np.zeros(image.size).reshape(image.shape)
    for xx in range(image.shape[0]):
        for yy in range(image.shape[1]):
            if (image[xx][yy] > threshold):
                output[xx][yy] = 1
    return output

def cal_nss(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred = pred / 255.0
    gt = gt / 255.0
    std = np.std(pred)
    u = np.mean(pred)

    smap = (pred - u) / (std + 1e-12)
    fixation_map = (gt - np.min(gt)) / (np.max(gt) - np.min(gt) + 1e-12)
    fixation_map = image_binary(fixation_map, 0.1)

    nss = smap * fixation_map

    nss = np.sum(nss) / np.sum(fixation_map+1e-12)

    return nss

def extract_bbox_points_think(output_text, x_factor, y_factor):
    pred_bboxes = []
    pred_points = []
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
    think_text = ""
    think_match = re.search(think_pattern, output_text)
    if think_match:
        think_text = think_match.group(1)

    rethink_pattern = r'<rethink>([^<]+)</rethink>'
    rethink_text = ""
    rethink_match = re.search(rethink_pattern, output_text)
    if rethink_match:
        rethink_text = rethink_match.group(1)
    
    return pred_bboxes, pred_points, think_text
    
#We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
reasoning_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "",
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        device_map="auto",
    )
        
segmentation_model = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
    
reasoning_model.eval()

# default processer
processor = AutoProcessor.from_pretrained("", padding_side="left")

        
QUESTION_TEMPLATE = \
            "Please answer \"{Question}\" with bboxs and points." \
            "Analyze the functional properties of specific parts of each object in the image and carefully find all the part(s) that matches the problem." \
            "Output the thinking process in <think> </think>, rethinking process in <rethink> </rethink> and final answer in <answer> </answer> tags." \
            "Output the bbox(es) and point(s) and affordance tpye(s) inside the interested object(s) in JSON format." \
            "i.e., <think> thinking process here </think>" \
            "<rethink> rethinking process here </rethink>" \
            "<answer>{Answer}</answer>"
    
   


# TODO: replace with model inference
def process_image(image_path, query) -> np.ndarray:
     
    image = PILImage.open(image_path)
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
                Question= query.lower().strip("."),
                Answer="[{\"bbox_2d\": [10,100,200,210], \"point_2d\": [30,110], \"affordance\": \"hold\"}, {\"bbox_2d\": [225,296,706,786], \"point_2d\": [302,410], \"affordance\": \"grasp\"}]"
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
    # print(points, len(points))
    
    # print("Thinking process: ", think)
    print(bboxes)
    # pdb.set_trace()
    
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        mask_all = np.zeros((image.height, image.width), dtype=bool)
        prob_all = np.zeros((image.height, image.width), dtype=np.float32)
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
            prob = masks[0]  # 获取概率分布
            prob_all = np.maximum(prob_all, prob)  # 逐元素取最大值
    print(mask)
    # image = Image.open(image_path).convert("RGB")
    # h, w = image.size
    # pred_map = np.random.randint(0, 256, (w, h), dtype=np.uint8)
    return  mask_all, prob_all

def main():
    args = argparse.ArgumentParser()
    args.add_argument('--dataset', type=str, choices=['agd', 'umd'], default='agd')
    # UMD-part-affordance/part-affordance-dataset for umd
    args.add_argument('--dataset_path', type=str, default='AGD20K/Seen/testset/egocentric')
    args = args.parse_args()

    i_list = []
    u_list = []
    iou_list = []
    kl_list = []
    sim_list = []
    nss_list = []

    # total_files = 0
    # for root, dirs, files in os.walk(args.dataset_path):
    #     if not any(os.path.isdir(os.path.join(root, d)) for d in dirs):
    #         for file in files:
    #             if file.endswith('.jpg'):
    #                 total_files +=1

    # agd: Seen-test split
    if args.dataset == 'agd':
        for root, dirs, files in os.walk(args.dataset_path):
            if not any(os.path.isdir(os.path.join(root, d)) for d in dirs):
                action = root.split('/')[-2]
                category = root.split('/')[-1]
                for file in files:
                    if not file.endswith('.jpg'):
                        continue
                    file_path = os.path.join(root, file)
                    gt_path =  file_path.replace('egocentric', 'GT').replace('.jpg', '.png')
                    # some images do not have gt
                    if not os.path.exists(gt_path):
                        continue
                    # query reference: Affordancellm
                    query = f"What part of the {category} should we interact with in order to {action} it?"

                    pred_map, prob = process_image(file_path, query)

                    gt_mask_pil = Image.open(gt_path)
                    # gt_mask: 0-255
                    gt_mask = np.array(gt_mask_pil)
                    # 统计 ground-truth 中大于 0 的像素个数
                    gt_pos = np.count_nonzero(gt_mask > 0)

                    # 统计 pred_map 中大于 0 的像素个数
                    pred_pos = np.count_nonzero(pred_map > 0)

                    # print(f'GT > 0: {gt_pos}')
                    # print(f'Pred > 0: {pred_pos}')
                    # print(file_path)
                    # exit(0)
                    gt_mask_normalized = gt_mask.astype(np.float32) / 255.0

                    iou, intersection, union = calc_255_array_iou(gt_mask, pred_map)
                    # pred_tensor = torch.from_numpy(prob)
                    # gt_tensor = torch.from_numpy(gt_mask_normalized)

                    kl = cal_kl( pred_map,gt_mask)
                    sim = cal_sim(pred_map, gt_mask)
                    nss = cal_nss(pred_map, gt_mask)

                    i_list.append(intersection)
                    u_list.append(union)
                    iou_list.append(iou)
                    kl_list.append(kl)
                    sim_list.append(sim)
                    nss_list.append(nss)     


    # umd: custom split (1 / 10 in manu-labeled (1 / 3) images) on all images
    elif args.dataset == 'umd':
        affordance_map = {
            # Cut类工具
            'knife': ['cut', 'grasp'],
            'saw': ['cut', 'grasp'],
            'scissors': ['cut', 'grasp'],
            'shears': ['cut', 'grasp'],

            # Scoop类工具
            'scoop': ['scoop', 'grasp'],
            'spoon': ['scoop', 'grasp'],
            'trowel': ['scoop', 'grasp'],

            # Contain类工具
            'bowl': ['contain', 'grasp'],
            'cup': ['contain', 'grasp'],
            'ladle': ['contain', 'grasp'],
            'mug': ['contain', 'grasp'],
            'pot': ['contain', 'grasp'],

            # Support类工具
            'shovel': ['support', 'grasp'],
            'turner': ['support', 'grasp'],

            # Pound类工具
            'hammer': ['pound', 'grasp'],
            'mallet': ['pound', 'grasp'],
            'tenderizer': ['pound', 'grasp']
        }
        afford_dict_name_to_num = {
            'grasp': 1,
            'cut': 2,
            'scoop': 3,
            'contain': 4,
            'pound': 5,
            'support': 6,
            'wrap-grasp': 7
        }
        # 计算符合条件的文件总数
        total_files = 0
        for root, dirs, files in os.walk(args.dataset_path):
            if not any(os.path.isdir(os.path.join(root, d)) for d in dirs):
                category = root.split('/')[-1].split('_')[0]
                actions = affordance_map[category]
                for file in files:
                    print(file)
                    if not file.endswith('.jpg'):
                        continue
                    file_parts = file.split('_')
                    if len(file_parts) < 2:
                        continue
                    file_index = int(file_parts[-2])
                    if file_index % 30 == 1:
                        total_files += 1

        with tqdm(total=total_files, desc="Processing files") as pbar:        
            for root, dirs, files in os.walk(args.dataset_path):
                if not any(os.path.isdir(os.path.join(root, d)) for d in dirs):
                    category = root.split('/')[-1].split('_')[0]
                    actions = affordance_map[category]
                    for file in files:
                        if not file.endswith('.jpg'):
                            continue
                        # 从文件名中提取索引，例如 bowl_01_00000188_rgb.jpg -> 188
                        file_parts = file.split('_')
                        file_index = int(file_parts[-2])
                        # 1 / 3 with manu-labeled affordance, 1 / 10 for test split -> 1 / 30 for test
                        if not file_index % 30 == 1:
                            continue 
                        file_path = os.path.join(root, file)
                        for action in actions:
                            query = f"What part of the {category} should we interact with in order to {action} it?"
                            pred_map = process_image(file_path, query)

                            gt_file_path = file_path.replace('_rgb.jpg', '_label_rank.mat')
                            mat_data = scipy.io.loadmat(gt_file_path)
                            gt_mat = mat_data['gt_label']

                            # 创建ground truth mask
                            action_index = afford_dict_name_to_num[action] - 1  # 需要定义反向映射
                            h, w = gt_mat.shape[:2]
                            gt_mask = np.zeros((h, w))
                            max_val = np.max(gt_mat)
                            for i in range(h):
                                for j in range(w):
                                    if gt_mat[i, j, action_index] == 0:
                                        gt_mask[i, j] = 0
                                    else:
                                        gt_mask[i, j] = 1 - gt_mat[i, j, action_index] / max_val
                            # 转换为0-255范围
                            gt_mask = (gt_mask * 255).astype(np.uint8)

                            iou, intersection, union = calc_255_array_iou(gt_mask, pred_map)

                            # kl = cal_kl(pred_map, gt_mask)
                            # sim = cal_sim(pred_map, gt_mask)
                            # nss = cal_nss(pred_map, gt_mask)

                            i_list.append(intersection)
                            u_list.append(union)
                            iou_list.append(iou)
                            # kl_list.append(kl)
                            # sim_list.append(sim)
                            # nss_list.append(nss)
                    pbar.update(1)  # 更新进度条

    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    
    print(f'giou: {np.mean(iou_list)}')
    print(f'ciou: {np.sum(i_list) / np.sum(u_list)}')
    # print(f'kl: {np.mean(kl_list)}')
    # print(f'sim: {np.mean(sim_list)}')
    # print(f'nss: {np.mean(nss_list)}')
 

    iou_scores = np.array(iou_list)
    p_at_50 = np.sum(iou_scores > 0.5) / len(iou_scores)
    thresholds = np.arange(0.5, 1.0, 0.05)
    p_values_over_thresholds = []
    for th in thresholds:
        # 在当前阈值下，计算成功检测的比例
        precision_at_th = np.sum(iou_scores > th) / len(iou_scores)
        p_values_over_thresholds.append(precision_at_th)
    # P@50:95 是所有这些比例的平均值
    p_at_50_95 = np.mean(p_values_over_thresholds)

    print(f'p_at_50: {p_at_50}')
    print(f'p_at_50_95: {p_at_50_95}')
       # 创建一个字典来存储这些数据
    # kl = np.mean(kl_list)
    # sim = np.mean(sim_list)
    # nss = np.mean(nss_list)

    # # 创建一个字典来存储这些数据
    # stats_data = {
    #     'kl': kl,
    #     'sim': sim,
    #     'nss': nss
    # }

    # # 指定要保存的JSON文件路径
    # json_file_path = 'stats_data_agd.json'

    # # 将字典写入JSON文件
    # with open(json_file_path, 'w', encoding='utf-8') as json_file:
    #     json.dump(stats_data, json_file, ensure_ascii=False, indent=4)


       # 创建一个字典来存储这些数据
    stats_data = {
        'giou':np.mean(iou_list),
        'ciou': np.sum(i_list) / np.sum(u_list),
        'p_at_50': p_at_50,
        'p_at_50_95': p_at_50_95
    }

    # 指定要保存的JSON文件路径
    json_file_path = 'stats_data_agd.json'

    # 将字典写入JSON文件
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(stats_data, json_file, ensure_ascii=False, indent=4)

    print(f"统计数据已保存到 {json_file_path}")


if __name__ == "__main__":
    main()