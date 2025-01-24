#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J MyJob
#SBATCH -o ./jobs/MyJob.%J.out
#SBATCH -e ./jobs/MyJob.%J.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=6
##SBATCH --constraint=[v100]
#SBATCH --time=48:00:00
#SBATCH --mem=256G

#run the application:
source ~/.bashrc_ibex
# module purge
conda activate pytorch-env
which python
python main_low_wavenumbers_torch.py












###################### srun /home/plotnips/anaconda3/envs/torch2/bin/python main_low_wavenumbers_torch.py
