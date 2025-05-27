import os
import cv2 
import numpy as np 
import torch 
import torch.nn as nn 
import random
import torch.nn.functional as F 
from pycocotools import mask 
from torch.utils.data import Dataset


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from grefer import G_REFER
from refer import REFER


class ReferSegDataset(Dataset):
    def __init__(
        self, 
        base_image_dir,
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        data_split="train",
    ):
        
        DATA_DIR = os.path.join(base_image_dir, "refer_seg")
        
        self.refer_seg_ds_list = refer_seg_data.split(
            "||"
        )  # ['refclef', 'refcoco', 'refcoco+', 'refcocog']

        self.refer_seg_data = {}
        
        for ds in self.refer_seg_ds_list:
            if ds == "refcocog":
                splitBy = "umd"
            else:
                splitBy = "unc"

            if ds == "grefcoco":
                refer_api = G_REFER(DATA_DIR, ds, splitBy)
            else:
                refer_api = REFER(DATA_DIR, ds, splitBy)
                
            # print(refer_api.availableSplits)
                
            ref_ids_train = refer_api.getRefIds(split=data_split)
            images_ids_train = refer_api.getImgIds(ref_ids=ref_ids_train)
            refs_train = refer_api.loadRefs(ref_ids=ref_ids_train)
            
            refer_seg_ds = {}
            refer_seg_ds["images"] = []
            loaded_images = refer_api.loadImgs(image_ids=images_ids_train)

            for item in loaded_images:
                item = item.copy()
                if ds == "refclef":
                    item["file_name"] = os.path.join(
                        DATA_DIR, "images/saiapr_tc-12", item["file_name"]
                    )
                else:
                    item["file_name"] = os.path.join(
                        DATA_DIR, "images/train2014", item["file_name"]
                    )
                refer_seg_ds["images"].append(item)
            
            # if ds == "grefcoco":
            #     refer_seg_ds["annotations"] = refer_api.loadRefs(ref_ids=ref_ids_train) # anns_train
            # else:
            refer_seg_ds["annotations"] = refer_api.Anns

            img2refs = {}
            for ref in refs_train:
                image_id = ref["image_id"]
                img2refs[image_id] = img2refs.get(image_id, []) + [
                    ref,
                ]
            refer_seg_ds["img2refs"] = img2refs
            self.refer_seg_data[ds] = refer_seg_ds
            
            print(
                "dataset {} (refs {}) (train split) has {} images and {} annotations and {} img2refs.".format(
                    ds,
                    splitBy,
                    len(refer_seg_ds["images"]),
                    len(refer_seg_ds["annotations"]),
                    len(refer_seg_ds["img2refs"]),
                )
            )
            
        self.refer_api = refer_api

    def __len__(self):
        return len(self.refer_seg_data.keys())  

    def __getitem__(self, index):
        ds = random.randint(0, len(self.refer_seg_ds_list) - 1)
        ds = self.refer_seg_ds_list[ds]
        refer_seg_ds = self.refer_seg_data[ds] 
        return 1 
    
    
if __name__ == "__main__":
    dataset = ReferSegDataset(base_image_dir="")
    