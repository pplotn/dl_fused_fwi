from imports import *
from utils import *
from data_loader import *
from data_loader import DataLoader
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)
if __name__ == '__main__':
    gan = Pix2Pix()
    # gan.train(epochs=2, batch_size=1, sample_interval=1000)
    gan.sample_images(0,1)
dasad=1   
