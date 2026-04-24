#!/bin/bash
#SBATCH -J eval-sig
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 8:00:00
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

# Pipeline (all in one script):
#   1. inference on 4 r=100 candidates → pick best as "r100"
#   2. inference on baseline (YOLO + system-level SMT + concat)
#   3. inference on r0, r025, r05 full-page checkpoints
#   4. write per-page wide CSV
#   5. paired Wilcoxon + Holm correction → markdown table


echo "=== eval_for_significance: 7 inference passes + paired Wilcoxon ==="
python eval_for_significance.py

echo "Job finished: $(date)"
