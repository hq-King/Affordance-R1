import re
import json
import math
import pdb

def seg_thinking_format_reward(predict_str: str) -> float:
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    match = re.fullmatch(pattern, predict_str, re.DOTALL)
    return 1.0 if match else 0.0

def seg_segmentation_format_reward(predict_str: str) -> float:
    def is_valid_format(predict_str: str) -> bool:
        try:
            json_match = re.search(r'{[^}]+}', predict_str)
            if not json_match:
                return False
            json_str = json_match.group(0)
            data = json.loads(json_str)
            
            # check the required keys
            required_keys = ['bbox', 'points_1', 'points_2']
            for key in required_keys:
                if key not in data:
                    return False
            
            # check the format of the value
            bbox = data['bbox']
            if not isinstance(bbox, list) or len(bbox) != 4:
                return False
                
            points_1 = data['points_1']
            points_2 = data['points_2']
            if not isinstance(points_1, list) or len(points_1) != 2:
                return False
            if not isinstance(points_2, list) or len(points_2) != 2:
                return False

            return True
        except Exception:
            return False
    return 1.0 if is_valid_format(predict_str) else 0.0

def seg_iou_reward(predict_str: str, ground_truth: str) -> float:
    def iou(box1, box2):
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[2], box2[2])
        inter_y2 = min(box1[3], box2[3])
        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
        else:
            inter = 0
        area1 = (box1[2]-box1[0]+1)*(box1[3]-box1[1]+1)
        area2 = (box2[2]-box2[0]+1)*(box2[3]-box2[1]+1)
        union = area1 + area2 - inter
        return float(inter)/union
    
    try:
        ground_truth = ground_truth.strip()
        gt_box_pattern = r'<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>'
        gt_match = re.search(gt_box_pattern, ground_truth)
        if gt_match:
            gt_bbox = [int(gt_match.group(1)), int(gt_match.group(2)), int(gt_match.group(3)), int(gt_match.group(4))]
            
        json_pattern = r'{[^}]+}'  
        json_match = re.search(json_pattern, predict_str)
        # pdb.set_trace()
        if json_match:
            data = json.loads(json_match.group(0))
            bbox_key = 'bbox'
            if bbox_key and len(data[bbox_key]) == 4:
                content_bbox = data[bbox_key]
                if iou(content_bbox, gt_bbox) > 0.5:
                    return 1.0
    except Exception:
        pass
    return 0.0


def seg_box_l1_reward(predict_str: str, ground_truth: str) -> float:
    def l1_distance(box1, box2):
        return (abs(box1[0]-box2[0]) + abs(box1[1]-box2[1]) + abs(box1[2]-box2[2]) + abs(box1[3]-box2[3])) / 4
    
    try:
        ground_truth = ground_truth.strip()
        gt_box_pattern = r'<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>'
        gt_match = re.search(gt_box_pattern, ground_truth)
        if gt_match:
            gt_bbox = [int(gt_match.group(1)), int(gt_match.group(2)), int(gt_match.group(3)), int(gt_match.group(4))]
            
        json_pattern = r'{[^}]+}'  
        json_match = re.search(json_pattern, predict_str)
        if json_match:
            data = json.loads(json_match.group(0))
            bbox_key = 'bbox'
            if bbox_key and len(data[bbox_key]) == 4:
                content_bbox = data[bbox_key]
                if l1_distance(content_bbox, gt_bbox) < 10:
                    return 1.0
    except Exception:
        pass
    return 0.0

def seg_point_l1_reward(predict_str: str, ground_truth: str) -> float:
    def points_in_box(point, bbox):
        return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]
    
    def points_distance(points1, points2):
        dist1 = math.sqrt((points1[0][0]-points2[0][0])**2 + (points1[0][1]-points2[0][1])**2) + \
                math.sqrt((points1[1][0]-points2[1][0])**2 + (points1[1][1]-points2[1][1])**2)
        
        dist2 = math.sqrt((points1[0][0]-points2[1][0])**2 + (points1[0][1]-points2[1][1])**2) + \
                math.sqrt((points1[1][0]-points2[0][0])**2 + (points1[1][1]-points2[0][1])**2)
        return min(dist1, dist2) / 2
        
    try: 
        gt_points_pattern = r'<points>\((\d+),(\d+)\),\((\d+),(\d+)\)</points>'
        gt_match = re.search(gt_points_pattern, ground_truth)
        if gt_match:
            gt_points = [[int(gt_match.group(1)), int(gt_match.group(2))], [int(gt_match.group(3)), int(gt_match.group(4))]]
            
        json_pattern = r'{[^}]+}' 
        json_match = re.search(json_pattern, predict_str)

        if json_match:
            data = json.loads(json_match.group(0))
            # find bbox key
            bbox_key = 'bbox'
            if bbox_key and len(data[bbox_key]) == 4:
                content_bbox = data[bbox_key]
            # find points key
            points_keys = ['points_1', 'points_2']  # get the first two points keys
            if len(points_keys) == 2:
                point1 = data[points_keys[0]]
                point2 = data[points_keys[1]]
                point1 = [int(point1[0]), int(point1[1])]
                point2 = [int(point2[0]), int(point2[1])]
                if points_in_box(point1, content_bbox) and points_in_box(point2, content_bbox):
                    if points_distance([point1, point2], gt_points) < 100:
                        return 1.0
    except Exception:
        pass  # Continue to next verification method if this fails
    return 0.0

def seg_strict_compute_score(predict_str: str, ground_truth: str) -> float:
    thinking_format_reward = seg_thinking_format_reward(predict_str)
    segmentation_format_reward = seg_segmentation_format_reward(predict_str)
    iou_reward = seg_iou_reward(predict_str, ground_truth)
    point_l1_reward = seg_point_l1_reward(predict_str, ground_truth)
    box_l1_reward = seg_box_l1_reward(predict_str, ground_truth)
    
    reward = iou_reward + thinking_format_reward + segmentation_format_reward + point_l1_reward + box_l1_reward
    return reward