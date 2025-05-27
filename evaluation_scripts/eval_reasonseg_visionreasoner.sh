#!/bin/bash

REASONING_MODEL_PATH="/tos-bjml-researcheval/wanghanqing/model/affordance_r1_600/global_step_111/actor/huggingface"

SEGMENTATION_MODEL_PATH="facebook/sam2-hiera-large"

MODEL_DIR=$(echo $REASONING_MODEL_PATH | sed -E 's/.*vision_zero_workdir\/(.*)\/actor\/.*/\1/')
#TEST_DATA_PATH="Ricky06662/ReasonSeg_test"
TEST_DATA_PATH="/tos-bjml-researcheval/wanghanqing/Datasets/affordance_r1_partname_test_525"


TEST_NAME=$(echo $TEST_DATA_PATH | sed -E 's/.*\/([^\/]+)$/\1/')
OUTPUT_PATH="./reasonseg_eval_results/${MODEL_DIR}/${TEST_NAME}"

NUM_PARTS=4
# Create output directory
mkdir -p $OUTPUT_PATH

# Run 8 processes in parallel
for idx in {0..3}; do
    export CUDA_VISIBLE_DEVICES=$idx
    python evaluation_scripts/evaluation_visionreasoner.py \
        --reasoning_model_path $REASONING_MODEL_PATH \
        --segmentation_model_path $SEGMENTATION_MODEL_PATH \
        --output_path $OUTPUT_PATH \
        --test_data_path $TEST_DATA_PATH \
        --idx $idx \
        --num_parts $NUM_PARTS \
        --batch_size 8 &
done

# Wait for all processes to complete
wait

python evaluation_scripts/calculate_iou.py --output_dir $OUTPUT_PATH