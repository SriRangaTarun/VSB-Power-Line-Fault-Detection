import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.layers import *
from keras.models import *
import pyarrow.parquet as pq

from keras import backend as K
from keras import optimizers
from sklearn.model_selection
from keras.callbacks import *
from keras import activations
from keras import regularizers
from keras import initializers
from keras import constraints
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split 

from numba import jit
from math import log, floor
from sklearn.neighbors import KDTree
from scipy.signal import periodogram, welch

from keras.engine import Layer
from keras.engine import InputSpec
from keras.objectives import categorical_crossentropy
from keras.objectives import sparse_categorical_crossentropy

N_SPLITS = 5
sample_size = 800000
    
