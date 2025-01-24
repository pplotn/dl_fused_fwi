import torch
# print(torch.cuda.is_available())
# exit()
#########   util imports
from functools import partial
import pickle
import pathlib
from typing import Tuple
import json;    from argparse import ArgumentParser
import fnmatch
import scipy
import random
import matplotlib
fontsize=16
params={'image.cmap': 'RdBu_r'}
matplotlib.rcParams.update(params)
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import Bbox, TransformedBbox, Affine2D
from matplotlib import  tight_bbox
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
#########   ml imports
from skimage.transform import resize as imresize
from glob import glob
from joblib import dump,load
from sklearn.metrics import r2_score
import sklearn
import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from PIL import Image
import multiprocessing
from scipy.ndimage.filters import gaussian_filter,gaussian_filter1d
import torch.utils as utils
import torch.nn.init as init
import torch.utils.data as data
import torchvision.utils as v_utils
import torchvision.datasets as dset
################
#import ignite.distributed as idist
#from ignite.contrib.metrics.regression import R2Score
#from ignite.engine import Engine
#from ignite.exceptions import NotComputableError
#import pytest



########
# sys.path.append('../..')
# from F_utils import F_smooth
# a=1
# from ../../F_utils import F_smooth
# from __future__ import print_function, division
# from scipy import integrate
# import scipy.misc
# # import tensorflow.keras as keras
# from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
# from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
# from keras.layers import BatchNormalization, Activation, ZeroPadding2D
# from keras.layers.advanced_activations import LeakyReLU
# from keras.layers.convolutional import UpSampling2D, Conv2D
# from keras.models import Sequential, Model
# from keras.optimizers import Adam
# from keras.datasets import mnist
# import datetime
# import sys
# from data_loader import DataLoader
# import numpy as np
# import os
# import tensorflow as tf

