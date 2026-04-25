#!/bin/bash
#SBATCH -J eval-sig
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=atl1-1-01-005-17-0,atl1-1-01-005-19-0
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
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"

echo "--- nvidia-smi (pre-run) ---"
nvidia-smi

echo "--- torch CUDA check ---"
python -c "import torch; print('torch', torch.__version__, 'cuda_runtime', torch.version.cuda); print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count()); print('device 0:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

# Pipeline (all in one script):
#   1. inference on baseline (YOLO + system-level SMT + concat)
#   2. inference on r0, r025, r05, r100 full-page checkpoints
#   3. write per-page wide CSV
#   4. paired Wilcoxon + Holm correction → markdown table

echo "=== eval_for_significance: 4 fullpage + 1 baseline inference passes + paired Wilcoxon ==="
python eval_for_significance.py

echo "--- nvidia-smi (post-run) ---"
nvidia-smi

echo "Job finished: $(date)"
