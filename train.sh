#!/bin/bash
# --- Resource related ---
#SBATCH --ntasks=1
#SBATCH -A sds-rise
#SBATCH -t 3-00:00:00 # Hour:Minute:Second
#SBATCH -p gpu # Partation type
#SBATCH --gres=gpu:a100:4 # Request one A100 GPU
#SBATCH --mem-per-cpu=128GB # CPU memory

# module load anaconda cuda cudnn

# conda activate video

# ========================
python -m torch.distributed.run \
--nproc_per_node=4 \
--nnodes=1 \
--node_rank=0 \
main.py \
--mixed_precision fp16 \
--gradient_checkpointing \
--checkpoints_total_limit=2 \
--learning_rate=3e-5 \
--resume_from_checkpoint latest