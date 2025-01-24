#!/bin/bash
# salloc --nodes=1 --time=24:00:00 --gres=gpu:v100:1 --mem=128G --partition=batch
# salloc --nodes=1 --time=4:00:00 --gres=gpu:v100:1 --mem=128G --partition=batch
# salloc --nodes=1 --time=12:00:00 --gres=gpu:v100:1 --mem=128G --partition=batch
# salloc --nodes=1 --time=4:00:00 --gres=gpu:1 --partition=batch
# srun ~/anaconda3/envs/t_env/bin/python pix2pix.py
source ~/.bashrc_ibex
# srun ~/.conda/envs/pix2pix/bin/python main_low_wavenumbers_torch.py
srun /home/plotnips/anaconda3/envs/torch/bin/python main_low_wavenumbers_torch_unet.py
# srun /home/plotnips/anaconda3/envs/torch_10_2/bin/python main_low_wavenumbers_torch.py
