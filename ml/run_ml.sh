#!/bin/bash
# salloc --nodes=1 --time=24:00:00 --gres=gpu:v100:1 --mem=128G --partition=batch
# salloc --nodes=1 --time=4:00:00 --gres=gpu:v100:1 --mem=128G --partition=batch
# salloc --nodes=1 --time=12:00:00 --gres=gpu:v100:1 --mem=128G --partition=batch
# salloc --nodes=1 --time=4:00:00 --gres=gpu:1 --partition=batch
# srun ~/anaconda3/envs/t_env/bin/python pix2pix.py
source ~/.bashrc_ibex
srun ~/.conda/envs/pix2pix/bin/python main_low_wavenumbers.py
