set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # replace it with your local file path

RUN_NAME=$(basename "$0" .sh)

python3 -m verl.trainer.main \
    config=training_scripts/seg_zero_3b.yaml \
    data.val_files=None \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.kl_loss_coef=5.0e-3 \
    worker.actor.optim.lr=1.0e-6 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=2 \
    worker.rollout.enable_chunked_prefill=false \
    worker.rollout.n=8 \
    trainer.experiment_name=${RUN_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.total_episodes=1 \
    trainer.save_checkpoint_path=./workdir/${RUN_NAME}