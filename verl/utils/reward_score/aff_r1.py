import re
import json
from scipy.optimize import linear_sum_assignment
import numpy as np

def aff_r1_score_format_reward(predict_str: str) -> float:
    pattern = r"<think>.*?</think>\s*<rethink>.*?</rethink>\s*<answer>.*?</answer>"
    match = re.fullmatch(pattern, predict_str, re.DOTALL)
    thinking_format_reward = 1.0 if match else 0.0 
    
    def segmentation_format(predict_str: str) -> float:
        segmentation_format_reward = 0.0
        try:
            json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
            if not json_match:
                return segmentation_format_reward
            data = json.loads(json_match.group(1))
            
            data_cnt = len(data)
            
            for item in data:
                cur_reward = 0.0

                if 'bbox_2d' in item:
                    bbox_2d = item['bbox_2d']
                    if isinstance(bbox_2d, list) and len(bbox_2d) == 4:
                        cur_reward += 1.0
                    
                if 'point_2d' in item:
                    point_2d = item['point_2d']
                    if isinstance(point_2d, list) and len(point_2d) == 2:
                        cur_reward += 1.0
                
                if 'affordance' in item:
                    # affordance = item['affordance']
                    # if isinstance(bbox_2d, list) and len(bbox_2d) == 4:
                    cur_reward += 1.0

                # if 'object_part' in item:
                #     # affordance = item['affordance']
                #     # if isinstance(bbox_2d, list) and len(bbox_2d) == 4:
                #     cur_reward += 1.0
                
                segmentation_format_reward += cur_reward / data_cnt
        except Exception:
            pass
        return segmentation_format_reward
        
    segmentation_format_reward = segmentation_format(predict_str)
    
    return thinking_format_reward + segmentation_format_reward

def aff_r1_score_accuracy_reward(predict_str: str, ground_truth: str) -> float:
    max_accuracy_reward = 0.0
    MAX_OBJECTS = 120  # 设置上限
    
    try:
        gt_data = json.loads(ground_truth)
        gt_bboxes = [item['bbox_2d'] for item in gt_data]
        gt_points = [item['point_2d'] for item in gt_data]
            
        #json_match = re.search(r'```json\s*(.*?)\s*```', predict_str, re.DOTALL)
        json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
            pred_bboxes = [item['bbox_2d'] for item in data]
            pred_points = [item['point_2d'] for item in data]
            
            # 只有当预测或真实值超过上限时才截断
            if len(pred_bboxes) > MAX_OBJECTS:
                pred_bboxes = pred_bboxes[:MAX_OBJECTS]
                pred_points = pred_points[:MAX_OBJECTS]
            
            if len(gt_bboxes) > MAX_OBJECTS:
                gt_bboxes = gt_bboxes[:MAX_OBJECTS]
                gt_points = gt_points[:MAX_OBJECTS]
            
            # 预处理数据为numpy数组
            pred_bboxes = np.array(pred_bboxes)  # (M,4)
            pred_points = np.array(pred_points)  # (M,2)
            gt_bboxes = np.array(gt_bboxes)    # (N,4)
            gt_points = np.array(gt_points)     # (N,2)
            
            # 并行计算所有指标
            iou_matrix = batch_iou(pred_bboxes, gt_bboxes)  # (M,N)
            l1_matrix = batch_l1_distance(pred_bboxes, gt_bboxes)  # (M,N)
            points_dist_matrix = batch_points_distance(pred_points, gt_points)  # (M,N)
            points_in_box = batch_points_in_box(pred_points, pred_bboxes)  # (M,)
            
            # 计算reward矩阵
            iou_reward = (iou_matrix > 0.65).astype(float)
            bbox_l1_reward = (l1_matrix < 10).astype(float)
            point_reward = ((points_dist_matrix < 30) & points_in_box[:,np.newaxis]).astype(float)
            
            # 构建最终的cost矩阵
            cost_matrix = 3.0 - (iou_reward + bbox_l1_reward + point_reward)
            
            # 使用匈牙利算法找最优匹配
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # 直接从cost_matrix计算总reward
            total_reward = len(row_indices) * 3.0 - cost_matrix[row_indices, col_indices].sum()
            
            # 计算平均reward
            max_length = max(len(pred_bboxes), len(gt_bboxes))
            max_accuracy_reward = total_reward / max_length
            
    except Exception:
        pass
    return max_accuracy_reward

def aff_r1_score_non_repeat_reward(predict_str: str) -> float:
    non_repeat_reward = 1.0  # 初始满分
    try:
        sentences = predict_str.split('.')
        
        # 移除空句子
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 检查重复
        seen = set()
        repeats = 0
        
        for sentence in sentences:
            if sentence in seen:
                repeats += 1
            if repeats >=2:
                non_repeat_reward = 0
                break
            seen.add(sentence)
            
    except Exception:
        pass
    
    return non_repeat_reward

def aff_reward_compute_score(predict_str:str, affordance_truth: str,sim_model):
    max_accuracy_reward = 0
    try:
        aff_name = affordance_truth
            
        #json_match = re.search(r'```json\s*(.*?)\s*```', predict_str, re.DOTALL)
        json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
            aff_pred = [item['affordance'] for item in data]
            sim = sim_model.similarity(aff_pred, aff_name)
            if sim > 0.8:
                max_accuracy_reward = 1
            
    except Exception:
        pass
    return max_accuracy_reward

def part_reward_compute_score(predict_str:str, affordance_truth: str,sim_model):
    max_accuracy_reward = 0
    try:
        # gt_data = json.loads(affordance_truth)
        part_name = affordance_truth
            
        #json_match = re.search(r'```json\s*(.*?)\s*```', predict_str, re.DOTALL)
        json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
            part_pred = [item['object_part'] for item in data]
            sim = sim_model.similarity(part_pred , part_name)
            if sim > 0.8:
                max_accuracy_reward = 1
            
    except Exception:
        pass
    return max_accuracy_reward


def aff_r1_score(predict_str: str, ground_truth: str,affordance_truth: str,part_truth: str,sim_model) -> float:
    # print(predict_str, ground_truth)
    # 新增：判断 bbox 个数是否与 gt 相同的奖励
    bbox_count_reward = 0.0  # 默认奖励为0

    try:
        # 提取模型输出的 bbox 个数
        json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
            pred_bboxes_count = len(data)
            
            # 提取 gt 的 bbox 个数
            gt_data = json.loads(ground_truth)
            gt_bboxes_count = len(gt_data)
            # print(gt_data)
            print(gt_bboxes_count )
            # 如果个数相同，奖励为1
            if pred_bboxes_count == gt_bboxes_count:
                bbox_count_reward = 1.0
    
    except Exception:
        pass
    
    format_reward = aff_r1_score_format_reward(predict_str)
    accuracy_reward = aff_r1_score_accuracy_reward(predict_str, ground_truth)
    non_repeat_reward = aff_r1_score_non_repeat_reward(predict_str)
    aff_reward = aff_reward_compute_score(predict_str,affordance_truth,sim_model)
    # part_reward = part_rewarf_compute_score(predict_str,part_truth,sim_model)
    # affordance_reward = 0
    # fruit = aff_reward + part_reward
    # if fruit == 2:
    #     affordance_reward  = 1
    # 添加一个新的奖励条件
    reward = format_reward + accuracy_reward + non_repeat_reward + aff_reward + bbox_count_reward
    return reward

def batch_iou(boxes1, boxes2):
    # boxes1: (M,4), boxes2: (N,4)
    # 广播机制自动扩展维度
    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)  # (M,1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)  # (N,1)
    
    xA = np.maximum(x11, np.transpose(x21))  # (M,N)
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    
    interArea = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)
    box1Area = (x12 - x11 + 1) * (y12 - y11 + 1)  # (M,1)
    box2Area = (x22 - x21 + 1) * (y22 - y21 + 1)  # (N,1)
    
    unionArea = box1Area + np.transpose(box2Area) - interArea
    iou = interArea / unionArea  # (M,N)
    return iou

def batch_l1_distance(boxes1, boxes2):
    # boxes1: (M,4), boxes2: (N,4)
    boxes1 = boxes1[:, np.newaxis, :]  # (M,1,4)
    boxes2 = boxes2[np.newaxis, :, :]  # (1,N,4)
    return np.mean(np.abs(boxes1 - boxes2), axis=2)  # (M,N)

def batch_points_distance(points1, points2):
    # points1: (M,2), points2: (N,2)
    points1 = points1[:, np.newaxis, :]  # (M,1,2)
    points2 = points2[np.newaxis, :, :]  # (1,N,2)
    
    # 计算欧氏距离
    dist = np.sqrt(np.sum((points1 - points2)**2, axis=2))  # (M,N)
    return dist

def batch_points_in_box(points, boxes):
    """
    检查每个点是否在对应的框内
    points: (M,2) - M个点的坐标
    boxes: (M,4) - M个框的坐标 [x1,y1,x2,y2]
    返回: (M,) 布尔数组
    """
    x_check = (points[:,0] >= boxes[:,0]) & (points[:,0] <= boxes[:,2])
    y_check = (points[:,1] >= boxes[:,1]) & (points[:,1] <= boxes[:,3])
    return x_check & y_check


if __name__ == "__main__":
    predict_str = """
<answer>
[{"bbox_2d": [10, 100, 398, 423], "point_2d": [283, 169]}]
</answer>
"""
    ground_truth = """
[{"bbox_2d": [416, 7, 833, 553], "point_2d": [648, 249]}]"""
    print(predict_str)
    print(ground_truth)
    print(aff_r1_score(predict_str, ground_truth))
