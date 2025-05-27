#!/bin/bash

REASONING_MODEL_PATH="Ricky06662/Seg-Zero-7B-Best-on-ReasonSegTest"
SEGMENTATION_MODEL_PATH="facebook/sam2-hiera-large"


OUTPUT_PATH="./reasonseg_eval_results"
TEST_DATA_PATH="Ricky06662/ReasonSeg_test"
# TEST_DATA_PATH="Ricky06662/ReasonSeg_val"
NUM_PARTS=8

# Create output directory
mkdir -p $OUTPUT_PATH

# Run 8 processes in parallel
for idx in {0..7}; do
    export CUDA_VISIBLE_DEVICES=$idx
    python evaluation_scripts/evaluation.py \
        --reasoning_model_path $REASONING_MODEL_PATH \
        --segmentation_model_path $SEGMENTATION_MODEL_PATH \
        --output_path $OUTPUT_PATH \
        --test_data_path $TEST_DATA_PATH \
        --idx $idx \
        --num_parts $NUM_PARTS \
        --batch_size 100 &
done

# Wait for all processes to complete
wait

python evaluation_scripts/calculate_iou.py --output_dir $OUTPUT_PATH