#!/bin/bash
#SBATCH -J stacked-precomp
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

echo "=== Starting pre-computed stacking curriculum training ==="
echo "  Dataset: StackingCropDataset (strict n-matching)"
echo "  Stages 2-5: effective batch=16, lr=2e-5"
echo "  Stages 6-9: effective batch=8,  lr=1e-5"
python train_precomputed_stacking.py \
    --config config/stacked_precomputed_9stage.gin

echo "Job finished: $(date)"
