#!/bin/bash
CUDA_VISIBLE_DEVICES=6
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python cli_demo.py \
    --model_path "../CogVideoX-2b" \
    --dtype "bfloat16" \
    --output_dir "/home/tuyijing/CogVideoX/finetune/results/Cogvideox-2b-base" \
    --output_mp4_file "a_dog.mp4" \
    --generate_type "t2v" \
    --prompt " A dog is the focal point against a gray backdrop. Initially, the dog's tongue is playfully out, and its ears are perked up, adding to its charming demeanor. As time passes, the dog's expressions and poses remain consistent, with the tongue occasionally protruding, suggesting a playful and whimsical atmosphere. The dog's attire and the plain background emphasize its unique style and the humorous contrast between its cool demeanor and the warm, inviting setting. The dog's alert and curious gaze occasionally shifts, maintaining the playful and whimsical mood." \


# "A lively dog races across a sunlit garden, weaving through tall grass and colorful blossoms. Its eyes shine with joy, its tongue hanging out as it enjoys the thrill of the run."
