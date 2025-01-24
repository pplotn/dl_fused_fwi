# source ~/.bashrc_ibex
# conda activate pix2pix
# conda deactivate
# conda remove -n pix2pix --all
# conda create --name pix2pix python=3.6
pip3 install -r ../requirements_orig.txt
conda install cudatoolkit==10.1.243
pip install tensorflow-gpu==2.2.0
# pip install scipy==1.1.0
conda install scipy==1.1.0