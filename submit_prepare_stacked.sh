#!/bin/bash
#SBATCH -J prep-stacked
#SBATCH -p ice-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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

mkdir -p logs

echo "=== Generating stacked page data (fold 0, N=1..9) ==="
# real_samples_per_n=1688 matches the number of real system images in train split
# syn_samples_per_n=1872 matches the number of synthetic system images in train split
python prepare_stacked_data.py \
    --system_data_path      "/home/hice1/jwang3180/scratch/Fullpage Jazzmus/Jazzmuss_Data/jazzmus_systems" \
    --synthetic_system_path "/home/hice1/jwang3180/scratch/Fullpage Jazzmus/Jazzmuss_Data/jazzmus_systems_syn" \
    --fullpage_data_path    "/home/hice1/jwang3180/scratch/Fullpage Jazzmus/Jazzmuss_Data/jazzmus_fullpage" \
    --fullpage_syn_path     "/home/hice1/jwang3180/scratch/Fullpage Jazzmus/Jazzmuss_Data/jazzmus_fullpage_syn" \
    --out_dir               "/home/hice1/jwang3180/scratch/Fullpage Jazzmus/Jazzmuss_Data/jazzmus_stacked" \
    --fold 0 \
    --max_n 9 \
    --real_samples_per_n 1688 \
    --syn_samples_per_n  1872 \
    --system_height 256 \
    --width_tolerance 0.15

echo "=== Data generation complete: $(date) ==="
echo ""
echo "Split counts:"
wc -l "/home/hice1/jwang3180/scratch/Fullpage Jazzmus/Jazzmuss_Data/jazzmus_stacked/splits/"*.txt
echo ""
echo "To train:"
echo "  sbatch submit_train_stacked.sh"

echo "Job finished: $(date)"
