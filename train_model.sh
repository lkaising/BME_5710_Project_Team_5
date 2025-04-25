#!/bin/bash
# Script to train the enhanced MRI super-resolution model

# Set environment variables for PyTorch
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# Configuration
CONFIG_PATH="./configs/base_config.yaml"
OUTPUT_DIR="./checkpoints/enhanced_model"
MODEL="willnet_se_deep"
BATCH_SIZE=4
EPOCHS=250
MID_CHANNELS=64
N_BLOCKS=10
LR=2e-4

# Create output directory
mkdir -p $OUTPUT_DIR

# Train the model with enhanced parameters
python src/train_refactored.py \
  --config $CONFIG_PATH \
  --output_dir $OUTPUT_DIR \
  --model $MODEL \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --mid_channels $MID_CHANNELS \
  --n_blocks $N_BLOCKS \
  --lr $LR \
  --gamma 0.6 \
  --step_size 80

echo "Training complete! The best model is saved at: $OUTPUT_DIR/${MODEL}_best.pth"