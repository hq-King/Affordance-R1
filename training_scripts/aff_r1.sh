set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

MODEL_PATH=/fs-computility/ResearchEval/shared/hub/Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path

RUN_NAME=$(basename "$0" .sh)

python3 -m verl.trainer.main \
    config=training_scripts/aff_r1.yaml \
    data.val_files=None \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.kl_loss_coef=1.0e-2 \
    worker.actor.optim.lr=1.0e-6 \
    worker.actor.micro_batch_size_per_device_for_update=8 \
    worker.actor.micro_batch_size_per_device_for_experience=16 \
    worker.rollout.enable_chunked_prefill=false \
    worker.rollout.n=8 \
    trainer.experiment_name=affordacneR1 \
    trainer.n_gpus_per_node=4 \
    trainer.total_episodes=3 \
    trainer.save_checkpoint_path=/tos-bjml-researcheval/wanghanqing/model/affordnacer1_5.22