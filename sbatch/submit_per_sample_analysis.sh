#!/bin/bash
#SBATCH -J per-sample-analysis
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
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

echo "=== Per-sample comparative analysis: baseline vs. masking r=1.00 ==="
python per_sample_analysis.py \
    --data_path     "/home/hice1/jwang3180/scratch/Fullpage Jazzmus/Jazzmuss_Data/jazzmus_pagecrop" \
    --replay_ckpt   "/home/hice1/jwang3180/scratch/Fullpage Jazzmus/Jazzmus_weights/replay_100percent/pagecrop_fold0_best-v2.ckpt" \
    --baseline_ckpt "/home/hice1/jwang3180/scratch/jazzmus/ISMIR-Jazzmus/weights/smt_sys_best/smt_pre_syn_medium.ckpt" \
    --yolo_path     "/home/hice1/jwang3180/scratch/jazzmus/ISMIR-Jazzmus/yolo_weigths/yolov11s_20241108.pt" \
    --fold          0 \
    --final_stage   9 \
    --system_height 256 \
    --out_dir       "/home/hice1/jwang3180/FullPageJazzOMR/analysis_out"

echo "Job finished: $(date)"
