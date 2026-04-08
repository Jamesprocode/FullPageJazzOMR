#!/bin/bash
#SBATCH -J eval-ckpts
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 2:00:00
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

echo "=== Evaluating all checkpoints (best only) ==="
python eval_checkpoints.py

echo "=== Evaluating all per-stage checkpoints ==="
python eval_checkpoints.py --best_only=False

echo "Job finished: $(date)"
