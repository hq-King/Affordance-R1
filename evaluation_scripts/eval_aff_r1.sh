#!/bin/bash

REASONING_MODEL_PATH="Affordance_r1_7b"

SEGMENTATION_MODEL_PATH="facebook/sam2-hiera-large"

MODEL_DIR=$(echo $REASONING_MODEL_PATH | sed -E 's/.*vision_zero_workdir\/(.*)\/actor\/.*/\1/')
#TEST_DATA_PATH="Ricky06662/ReasonSeg_test"
TEST_DATA_PATH="Affordance-R1/test"


TEST_NAME=$(echo $TEST_DATA_PATH | sed -E 's/.*\/([^\/]+)$/\1/')
OUTPUT_PATH="Affordance_r1/reason_eval_results/${TEST_NAME}"

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