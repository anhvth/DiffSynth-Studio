# export CUDA_VISIBLE_DEVICES="3,4,5,6" 
# export CUDA_VISIBLE_DEVICES="1"
export NCCL_DEBUG=INFO
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1


export CUDA_VISIBLE_DEVICES="1,2,3,4,5,6,7"
ROOT=$(pwd)/
cd examples/ExVideo/
PYTHON=/home/anhvth5/miniconda3/envs/py39-ac/bin/python
$PYTHON -u ExVideo_svd_train_custom.py \
  --pretrained_path $ROOT"models/stable_video_diffusion/svd_xt.safetensors" \
  --dataset_path $ROOT"datasets/tiktokdance" \
  --output_path $ROOT"outputs/models_tiktokdance" \
  --steps_per_epoch 8000 \
  --num_frames 24 \
  --height 512 \
  --width 320 \
  --dataloader_num_workers 2 \
  --learning_rate 1e-5 \
  --max_epochs 10 \
  --accumulate_grad_batches 4