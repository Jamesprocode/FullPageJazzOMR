#!/bin/bash
#SBATCH -J fullpage-cl
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH -t 16:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jwang3180@gatech.edu
#SBATCH -o logs/slurm_%j.out
#SBATCH -e logs/slurm_%j.err

cd /home/hice1/jwang3180/FullPageJazzOMR

source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate jazzmus

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"
nvidia-smi

echo "=== Running test_generators sanity check ==="
python test_generators.py \
    --data_path "/home/hice1/jwang3180/scratch/Fullpage Jazzmus/Jazzmuss_Data/jazzmus_pagecrop" \
    --fold 0 \
    --max_n 9

echo "=== Starting training ==="
python train.py \
    --config config/pagecrop_9stage_replay.gin

echo "Job finished: $(date)"
