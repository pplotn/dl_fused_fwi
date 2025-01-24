# import m8r as sf
import json;    from argparse import ArgumentParser
import os,sys
import glob
from itertools import islice, tee
import subprocess
import segyio
import pickle
import zipfile
from collections import Counter
import datetime
import math
import fnmatch
import skimage
from skimage.transform import resize as imresize
from skimage.transform import resize as resize
from matplotlib.gridspec import GridSpec
import random
import numpy as np
import numpy
import io
import matplotlib
params={'image.cmap': 'RdBu_r',
        'text.latex.preamble': [r"""\usepackage{bm}"""],
        'mathtext.fontset':'custom',}
        # 'mathtext.it':'STIXGeneral:italic',
        # 'mathtext.bf':'STIXGeneral:italic:bold'}
matplotlib.rcParams.update(params)
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.transforms import Bbox, TransformedBbox, Affine2D
from matplotlib import  tight_bbox
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy.matlib as npm
import pandas as pd
import itertools
from itertools import islice
# import more_itertools
from collections import deque
import h5py
import scipy
from scipy import signal
from scipy.ndimage.filters import gaussian_filter,gaussian_filter1d
from scipy.ndimage.interpolation import map_coordinates
from scipy.fftpack import fft2, ifft2
from collections import namedtuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from numpy.random import randint as randint
#############################
#########  torch ml imports
import pandas as pd
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
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary
from PIL import Image
import multiprocessing
from scipy.ndimage.filters import gaussian_filter,gaussian_filter1d
import torch.utils as utils
import torch.nn.init as init
import torch.utils.data as data
import torchvision.utils as v_utils
import torchvision.datasets as dset
#####   rock physics
import bruges as br
from welly import Well