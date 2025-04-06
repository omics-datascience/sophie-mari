#!/bin/bash
#SBATCH --job-name=gaknn-array
#SBATCH --output=slurm/gaknn_logs/output_%A_%a.out
#SBATCH --error=slurm/gaknn_logs/error_%A_%a.err
#SBATCH --array=0-11
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Load your environment
source ~/.bashrc
conda activate gaknn-env

# Navigate to project directory
cd /network/rit/lab/fuchslab/SophiaB/sophie-mari

# Create logs directory if it doesn't exist (this is redundant here but harmless)
mkdir -p slurm/gaknn_logs

# Compute drug index range based on SLURM_ARRAY_TASK_ID
start_idx=$((SLURM_ARRAY_TASK_ID * 50))
end_idx=$((start_idx + 50))

# Run the Python script on this chunk
python scr/gaknn_slurm.py $start_idx $end_idx