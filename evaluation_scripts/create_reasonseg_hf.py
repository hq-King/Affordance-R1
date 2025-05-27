from datasets import Dataset, DatasetDict, Image, Features, Value, Array2D, Sequence
from huggingface_hub import create_repo, HfApi
import os
import json
from PIL import Image as PILImage
import glob
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_mask_from_json(json_path, height, width):
    try:
        with open(json_path, "r") as r:
            anno = json.loads(r.read())
    except:
        with open(json_path, "r", encoding="cp1252") as r:
            anno = json.loads(r.read())

    inform = anno["shapes"]

    ### sort polies by area
    area_list = []
    valid_poly_list = []
    for i in inform:
        label_id = i["label"]
        points = i["points"]
        if "flag" == label_id.lower():  ## meaningless deprecated annotations
            continue

        tmp_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.polylines(tmp_mask, np.array([points], dtype=np.int32), True, 1, 1)
        cv2.fillPoly(tmp_mask, np.array([points], dtype=np.int32), 1)
        tmp_area = tmp_mask.sum()

        area_list.append(tmp_area)
        valid_poly_list.append(i)

    ### ground-truth mask
    sort_index = np.argsort(area_list)[::-1].astype(np.int32)
    sort_index = list(sort_index)
    sort_inform = []
    for s_idx in sort_index:
        sort_inform.append(valid_poly_list[s_idx])

    mask = np.zeros((height, width), dtype=np.uint8)
    for i in sort_inform:
        label_id = i["label"]
        points = i["points"]

        if "ignore" in label_id.lower():
            label_value = 255  # ignored during evaluation
        else:
            label_value = 1  # target

        cv2.polylines(mask, np.array([points], dtype=np.int32), True, label_value, 1)
        cv2.fillPoly(mask, np.array([points], dtype=np.int32), label_value)

    mask = mask.astype(bool)  
    return mask


def create_and_push_dataset(data_dir, test_data, repo_id, data_split):
    """
    Create a dataset and push it to the Hub
    """
    # 1. Process data
    def process_split(split_data):
        processed_data = split_data.copy()
        processed_data['image'] = [
            os.path.join(data_dir, img_path) 
            for img_path in split_data['image']
        ]
        return processed_data
    
    # 2. Create a dataset (only contains test set)
    dataset = Dataset.from_dict(
            process_split(test_data),
            features=Features({
                'image': Image(),
                'mask': Sequence(Sequence(Value('bool'))),
                'text': Value('string'),
                'image_id': Value('string'),
                'ann_id': Value('string'),
                'img_height': Value('int64'),
                'img_width': Value('int64')
            })
        )
    
    # 3. Push to the Hub
    dataset.push_to_hub(
        repo_id,
        split="test"
    )
    
    print(f"Dataset pushed to: https://huggingface.co/datasets/{repo_id}")
    return dataset

# Example usage
if __name__ == "__main__":
    data_split = "test"
    # data_split = "val"
    data_dir = f"/gpfs/yuqiliu/data/Segmentation/reason_seg/{data_split}"
    json_path_list = sorted(glob.glob(data_dir + "/*.json"))
    
    image_list = []
    mask_list = []
    text_list = []
    image_id_list = [] 
    ann_id_list = [] 
    img_height_list = []
    img_width_list = []
    
    for idx, json_path in tqdm(enumerate(json_path_list)):
        img_path = json_path.replace(".json", ".jpg")
        image_list.append(img_path)
        text_list.append(json.loads(open(json_path, "r").read())["text"][0])
        id = json_path.split("/")[-1].split(".")[0]
        image_id_list.append(id)
        ann_id_list.append(id)
        width, height = PILImage.open(img_path).size
        img_height_list.append(height)
        img_width_list.append(width)
        mask_list.append(get_mask_from_json(json_path, height, width))
        # if idx > 5:
        #     break
        
    # Create test set data
    test_data = {
        'image': image_list,
        'text': text_list,
        'mask': mask_list,
        'image_id': image_id_list,
        'ann_id': ann_id_list,
        'img_height': img_height_list,
        'img_width': img_width_list
    }
    
    # Set Hugging Face repository ID
    repo_id = f"Ricky06662/ReasonSeg_{data_split}"  # Replace with your repository ID
    
    dataset = create_and_push_dataset(
        data_dir=data_dir,
        test_data=test_data,
        repo_id=repo_id,
        data_split=data_split
    )