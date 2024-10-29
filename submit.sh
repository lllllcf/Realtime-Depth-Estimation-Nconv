#!/bin/bash
# Request a GPU partition node and access to GPU
#SBATCH --partition=3090-gcondo
#SBATCH -p 3090-gcondo --gres=gpu:1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --mem=64G
#SBATCH -t 20:00:00
#SBATCH --output=./logs/train_step_1_void

module load cuda
module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate bp3

python void_train_step1.py