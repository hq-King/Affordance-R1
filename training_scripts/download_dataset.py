from datasets import load_dataset
import os

dataset_name= "Ricky06662/VisionReasoner_multi_object_1k_840"

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset(dataset_name)

# 创建数据保存目录
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", dataset_name.split('/')[-1])
os.makedirs(data_dir, exist_ok=True)

# 保存数据集到本地
ds.save_to_disk(data_dir)
print(f"Dataset save to: {data_dir}")