import os
import numpy
import pandas
import time
import sys                                      #to write out stuff printed to screen
import pickle
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.visualize_util import plot
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping,LearningRateScheduler
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import maxabs_scale
from sklearn.model_selection import GridSearchCV

# Define Constants
data_directory = '/storage1/users/jtr6/UCI_paper_data_sample/'
training_data_sample = 'not1000_train.npy'
test_data_sample = 'not1000_test.npy'
scaler = 'maxabs'                                                       #Options: 'none', 'maxabs', 'robust_scale' 
feature = 27
number_of_loops = 2                                                     #Total number of loops, is incremented later for functions who's index start at 0
number_of_epochs = 1                                                    #Just what it says, number of epochs never re-indexed
set_batch_size = 10000                                                  #Select batch size

# Fix random seed for reproducibility
seed = 42
numpy.random.seed(seed)


# Load UCI data
dataset = numpy.load(data_directory + training_data_sample)                     #Load data from numpy array
testset = numpy.load(data_directory + test_data_sample)                         #Load data from numpy array
