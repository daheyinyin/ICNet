#!/bin/bash

# ============================================================================

model_path=$1
framework=1
output_model_name=$2

atc \
    --model=$model_path \
    --framework=$framework \
    --output=$output_model_name \
    --input_format=NHWC \
    --input_shape="actual_input_1:1,1024,2048,3" \
    --log=error \
    --output_type=FP32 \
    --soc_version=Ascend310
