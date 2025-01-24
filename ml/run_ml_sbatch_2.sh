#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J MyJob
#SBATCH -o MyJob.%J.out
#SBATCH -e MyJob.%J.err
##SBATCH --gpus-per-node=gpu:v100:1
#SBATCH --gres=gpu:4
#SBATCH --constraint=[v100]
#SBATCH --cpus-per-gpu=6
#SBATCH --mail-type=ALL
#SBATCH --time=24:00:00
#SBATCH --mem=256G

##SBATCH --time=2:00:00
##SBATCH --mem=16G
##SBATCH --time=2:00:00
##SBATCH --mem=256G

#run the application:
source ~/.bashrc_ibex
module purge
conda activate torch2
which python
# nvidia-smi
python main_low_wavenumbers_torch.py

# srun /home/plotnips/anaconda3/envs/torch/bin/python main_low_wavenumbers_torch.py
# source /ibex/scratch/plotnips/intel/MWE/Keras-GAN/pix2pix/torch_venv/bin/activate
# which python
# srun python main_low_wavenumbers_torch.py
# srun /home/plotnips/anaconda3/envs/torch/bin/python clean_log_folders_delete_torch_models.py
