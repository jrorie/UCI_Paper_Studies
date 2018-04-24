from __future__ import print_function

from hyperopt import Trials, STATUS_OK, tpe
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils

import os
import numpy
import pandas
import time
import sys                     #to write out stuff printed to screen
import pickle
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

#from keras.models import Sequential
#from keras.layers import Dense, Dropout
from keras.utils.vis_utils import plot_model as plot
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


from hyperas import optim
from hyperas.distributions import choice, uniform, conditional

# Bring in and set global variables must be done before macros!
import globals
globals.init()

# Bring in external macros
import macros_benchmarks
from macros_benchmarks import *



# Create directory structure
if not os.path.exists('./plots/'):
    os.makedirs('./plots/')
if not os.path.exists('./saved_models/'):
    os.makedirs('./saved_models/')
if not os.path.exists('./logs/'):
    os.makedirs('./logs/')
if not os.path.exists('./predictions/'):
    os.makedirs('./predictions/')
if not os.path.exists('./roc_info/'):
    os.makedirs('./roc_info/')


#Set the start time for the entire process
overall_start_time = time.time()

#Open the log file
file = open('./logs/logFile_gridsearch.txt', 'w')

# Define Constants
data_directory = '/home/rice/jrorie/data/'
training_data_sample = 'not1000_train.npy'
test_data_sample = 'not1000_test.npy'
scaler = 'maxabs'
feature = 27
number_of_loops = 1                            #Total number of loops, is incremented later for functions who's index start at 0
number_of_epochs = 1                            #Just what it says, number of epochs never re-indexed
set_batch_size = 10000                            #Select batch size

# Fix random seed for reproducibility
seed = 42
numpy.random.seed(seed)

# Log Constants
file.write('--------------------------------\n')
file.write('    Definitions of Constants    \n')
file.write('--------------------------------\n')
file.write('Directory: %s\n'        % data_directory)
file.write('Training data: %s\n'    % training_data_sample)
file.write('Test data: %s\n'        % test_data_sample)
file.write('Seed value: %d\n'       % seed)
file.write('Feature number: %d\n'   % feature)
file.write('Number of loops: %d\n'  % number_of_loops)
file.write('Number of epochs: %d\n' % number_of_epochs)
file.write('Batch Size: %d\n'       % set_batch_size)
file.write('********************************\n')
file.write('********************************\n')



def data():
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """
    # Load UCI data
    dataset = numpy.load('/home/rice/jrorie/data/not1000_train.npy')                     #Load data from numpy array
    testset = numpy.load('/home/rice/jrorie/data/not1000_test.npy')                          #Load data from numpy array


    # Split into input (X) and output (Y) variables
    x_train_prescale = dataset[:,1:]
    y_train = dataset[:,0]
    x_test_prescale = testset[:,1:]
    y_test = testset[:,0]
    x_train, x_test = scale_x(x_train_prescale, x_test_prescale, 'maxabs')
    #globals.input_scale = x_train.shape[1]
    return x_train, y_train, x_test, y_test


def create_model(x_train, y_train, x_test, y_test):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
    model = Sequential()
    model.add(Dense(512, input_shape=(28,)))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([256, 512, 1024])}}))
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.add(Dropout({{uniform(0, 1)}}))

    # If we choose 'four', add an additional fourth layer
    if conditional({{choice(['three', 'four'])}}) == 'four':
        model.add(Dense(100))

        # We can also choose between complete sets of layers

        model.add({{choice([Dropout(0.5), Activation('linear')])}})
        model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                  optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})

    model.fit(x_train, y_train,
              batch_size={{choice([64, 128])}},
              epochs=1,
              verbose=2,
              validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}



def create_model_CNN(x_train, y_train, x_test, y_test):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
    input_model = Sequential()
    input_model.add(Input(shape=(height, width, depth)))                                                         #Input
    input_model.add(Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu'))    #Conv 1
    input_model.add(Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu'))    #Conv 2
    input_model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))                                                #Pool 1
    input_model.add(Dropout(drop_prob_1))                                                                        #Drop 1
    input_model.add(Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu'))    #Conv 3
    input_model.add(Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu'))    #Conv 4
    input_model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))                                                #Pool 2
    input_model.add(Dropout(drop_prob_1))                                                                        #Drop 2
    input_model.add(flat = Flatten())                                                                            #Flat
    input_model.add(Dense(hidden_size, activation='relu'))                                                        #Hidden
    input_model.add(Dropout(drop_prob_2))                                                                        #Drop 3
    input_model.add(Dense(num_classes, activation='softmax'))                                                    #Output

    input_model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy', 'mse']) # reporting the accuracy

    input_model.fit(X_train, Y_train,                # Train the model using the training set...
          batch_size=batch_size, epochs=num_epochs,
          verbose=1, validation_split=0.1) # ...holding out 10% of the data for validation
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': input_model}



if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
