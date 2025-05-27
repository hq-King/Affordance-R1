from datasets import load_from_disk, DatasetDict, concatenate_datasets
from huggingface_hub import create_repo, HfApi
import os
import pdb


# 设置Hub仓库名称
repo_name = "Ricky06662/coco_val"
# 加载数据集
dataset = load_from_disk(f"/gpfs/yuqiliu/data/{repo_name}")

# 创建仓库
try:
    create_repo(
        repo_name,
        repo_type="dataset",
        private=False
    )
    print(f"Create public repo: {repo_name}")
except Exception as e:
    print(f"Repo may already exist: {e}")

# 推送到Hub
dataset.push_to_hub(repo_name, private=False)
print(f"Dataset uploaded to: {repo_name}")

# 打印数据集信息
print("\nDataset info:")
print(dataset)