#!/usr/bin/env python3
import os
import shutil
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from transformers import AutoModelForCausalLM, AutoTokenizer

def push_to_hub(
    model_path: str,
    repo_name: str,
    private: bool = False
):
    """
    将模型推送到 Hugging Face Hub
    
    Args:
        model_path: 本地模型路径
        repo_name: Hugging Face仓库名称 (格式: username/repo-name)
        kl_coef: KL系数
        learning_rate: 学习率
        private: 是否为私有仓库
    """
    # 检查环境变量
    if "HUGGING_FACE_HUB_TOKEN" not in os.environ:
        raise ValueError("请设置 HUGGING_FACE_HUB_TOKEN 环境变量")

    # 初始化 HF API
    api = HfApi()
    
    # 创建仓库（如果不存在）
    try:
        create_repo(repo_name, private=private, exist_ok=True)
    except Exception as e:
        print(f"创建仓库时出错: {e}")
        return

    # 创建模型卡片
    readme_content = f"""
# VisionReasoner-7B


Code: https://github.com/dvlab-research/VisionReasoner

## Description

This is a VisionReasoner-7B model.  It introduces a decoupled architecture consisting of a reasoning model and a segmentation model. The reasoning model interprets user intentions, generates explicit reasoning chains, and produces positional prompts, which are subsequently used by the segmentation model to generate pixel-level masks.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# load model
model = AutoModelForCausalLM.from_pretrained("Ricky06662/VisionReasoner-7B")
tokenizer = AutoTokenizer.from_pretrained("Ricky06662/VisionReasoner-7B")
```
"""

    # 创建临时目录
    tmp_dir = Path("tmp_model_upload")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    try:
        # 复制模型文件
        shutil.copytree(model_path, tmp_dir, dirs_exist_ok=True)
        
        # 写入 README.md
        with open(tmp_dir / "README.md", "w", encoding="utf-8") as f:
            f.write(readme_content)

        # 推送到 Hub
        api.upload_folder(
            folder_path=str(tmp_dir),
            repo_id=repo_name,
            commit_message=f"Upload model files"
        )
        
        print(f"模型已成功推送到: https://huggingface.co/{repo_name}")
        
    finally:
        # 清理临时目录
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

def main():
    parser = argparse.ArgumentParser(description="将模型推送到 Hugging Face Hub")
    parser.add_argument("--model_path", type=str, required=True,
                      help="本地模型路径")
    parser.add_argument("--repo_name", type=str, required=True,
                      help="Hugging Face仓库名称 (格式: username/repo-name)")
    parser.add_argument("--private", action="store_true",
                      help="是否创建私有仓库")

    args = parser.parse_args()
    
    push_to_hub(
        model_path=args.model_path,
        repo_name=args.repo_name,
        private=args.private
    )

# Usage:
# python push_to_hub.py \
#     --model_path "/path/to/your/model" \
#     --repo_name "your-username/model-name"
if __name__ == "__main__":
    main()