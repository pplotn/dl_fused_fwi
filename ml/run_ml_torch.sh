#!/bin/bash
# salloc --nodes=1 --time=4:00:00 --gres=gpu:v100:1 --mem=128G --partition=batch
# salloc --nodes=1 --time=4:00:00 --gres=gpu:v100:1 --mem=256G --partition=batch
# salloc --nodes=1 --time=4:00:00 --gres=gpu:v100:1 --mem=64G --partition=batch
# salloc --nodes=1 --time=24:00:00 --gres=gpu:1 --mem=128G --partition=batch
# salloc --nodes=1 --time=24:00:00 --gres=gpu:v100:1 --mem=128G --partition=batch
# salloc --nodes=1 --time=12:00:00 --gres=gpu:v100:1 --mem=128G --partition=batch
# salloc --nodes=1 --time=48:00:00 --gres=gpu:v100:1 --mem=128G --partition=batch
# salloc --nodes=1 --time=4:00:00 --gres=gpu:1 --partition=batch
# salloc --nodes=1 --time=48:00:00 --gres=gpu:rtx2080ti:1 --mem=128G --partition=batch
# srun ~/anaconda3/envs/t_env/bin/python pix2pix.py

srun /home/plotnips/anaconda3/envs/torch/bin/python main_low_wavenumbers_torch.py

# source /ibex/scratch/plotnips/intel/MWE/Keras-GAN/pix2pix/torch_venv/bin/activate
# srun python main_low_wavenumbers_torch.py
which python

# srun /ibex/scratch/plotnips/intel/MWE/Keras-GAN/pix2pix/torch_venv/bin/activate/bin/python main_low_wavenumbers_torch.py
# srun /home/plotnips/anaconda3/envs/torch/bin/python clean_log_folders_delete_torch_models.py