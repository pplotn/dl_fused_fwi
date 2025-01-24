from __future__ import print_function, division
import scipy
from scipy import integrate
import scipy.misc
# import tensorflow.keras as keras
from skimage.transform import resize as imresize
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.datasets import mnist
import datetime
import matplotlib
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.transforms import Bbox, TransformedBbox, Affine2D
from matplotlib import  tight_bbox
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
from data_loader import DataLoader
import numpy as np
import os
import tensorflow as tf
from glob import glob
import fnmatch
import random
from joblib import dump,load
from sklearn.metrics import r2_score