# salloc --nodes=1 --time=24:00:00 --gres=gpu:v100:1 --mem=128G --partition=batch
# salloc --nodes=1 --time=4:00:00 --gres=gpu:v100:1 --mem=128G --partition=batch
# salloc --nodes=1 --time=8:00:00 --gres=gpu:v100:1 --mem=128G --partition=batch
# salloc --nodes=1 --time=4:00:00 --gres=gpu:1 --partition=batch
srun ~/anaconda3/envs/t_env/bin/python pix2pix.py
