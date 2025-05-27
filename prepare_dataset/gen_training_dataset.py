from datasets import Dataset, DatasetDict, Image, Features, Value
from huggingface_hub import create_repo
import os
import json
from tqdm import tqdm
import glob
from tqdm import tqdm
from PIL import Image as PILImage
import cv2


def scale_box_coordinates(bbox_2d, x_factor, y_factor):
    """
    对边界框坐标进行缩放
    
    bbox_2d: [x1, y1, x2, y2]
    """
    # 缩放边界框坐标
    scaled_bbox = [
        int(bbox_2d[0] * x_factor + 0.5),  # x1
        int(bbox_2d[1] * y_factor + 0.5),  # y1
        int(bbox_2d[2] * x_factor + 0.5),  # x2
        int(bbox_2d[3] * y_factor + 0.5)   # y2
    ]

    
    return scaled_bbox

def scale_point_coordinates(point_2d, x_factor, y_factor):
    """
    对中心点坐标进行缩放
    point_2d: [x, y]
    """

    # 缩放中心点坐标
    scaled_point = [
        int(point_2d[0] * x_factor + 0.5),  # x
        int(point_2d[1] * y_factor + 0.5)   # y
    ]
    
    return scaled_point

def create_local_dataset(train_data, output_dir, image_resize):
    """
    创建数据集并保存到本地
    """
    # 2. 处理数据
    def process_split(split_data, image_resize):
        processed_data = split_data.copy()
        images = []
        for img_path in split_data['image']:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (image_resize, image_resize), interpolation=cv2.INTER_AREA)
            images.append(img)
        
        processed_data['image'] = images
        return processed_data
    
    # 3. 创建数据集 (修改后只包含train集)
    dataset = DatasetDict({
        'train': Dataset.from_dict(
            process_split(train_data, image_resize),
            features=Features({
                'id': Value('string'),
                'problem': Value('string'),
                'solution': Value('string'),
                'image': Image(),
                'img_height': Value('int64'),
                'img_width': Value('int64')
            })
        )
    })
    
    # 4. 保存到本地
    dataset.save_to_disk(output_dir)
    print(f"数据集已保存到: {output_dir}")
    
    return dataset

# 使用示例
if __name__ == "__main__":
    data_path_list = [
        "prepare_dataset/seg_zero_reasonseg_annotation_list_all_various_item.json"
    ]

    data = []
    for data_path in data_path_list:
        data.extend(json.load(open(data_path, 'r')))
    
    image_resize = 840
        
    id_list = []
    problem_list = []
    solution_list = []
    image_list = []
    img_height_list = []
    img_width_list = []

    for idx, item in tqdm(enumerate(data)):
        id_list.append(item['id'])
        problem_list.append(item['problem'])
        
        image_list.append(item['image_path'])
        
        # print(item['image_path'])
        image = cv2.imread(item['image_path'])
        height, width = image.shape[:2]
        
        img_height_list.append(height)
        img_width_list.append(width)
        
        x_factor = 840 / width
        y_factor = 840 / height
        solution = []
        for box_idx in range(len(item['bboxes'])):
            solution.append({
                "bbox_2d": scale_box_coordinates(item['bboxes'][box_idx], x_factor, y_factor), # [x1, y1, x2, y2]
                "point_2d": scale_point_coordinates(item['center_points'][box_idx], x_factor, y_factor) # [x, y]
                
            })
        solution_list.append(json.dumps(solution))
        # if idx > 20:
        #     break


    train_data = {
        'id': id_list,
        'problem': problem_list,
        'solution': solution_list,
        'image': image_list,
        'img_height': img_height_list,
        'img_width': img_width_list
    }
    
    dataset = create_local_dataset(
        train_data=train_data,
        output_dir=f"data/SegZero_multi_object_reasonseg_{image_resize}", # input your own output path
        image_resize=image_resize
    )