import os
import numpy
import pandas
import time
import sys 					#to write out stuff printed to screen
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

import macros
from macros import *

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
file = open('./logs/logFile_training.txt', 'w')

# Define Constants
data_directory = '/storage1/users/jtr6/UCI_paper_data_sample/'
training_data_sample = 'not1000_train.npy'
test_data_sample = 'not1000_test.npy'
feature = 27
number_of_loops = 2							#Total number of loops, is incremented later for functions who's index start at 0
number_of_epochs = 1							#Just what it says, number of epochs never re-indexed
set_batch_size = 10000							#Select batch size

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
file.write('Seed value: %d]n'       % seed)
file.write('Feature number: %d\n'   % feature)
file.write('Number of loops: %d\n'  % number_of_loops)
file.write('Number of epochs: %d\n' % number_of_epochs)
file.write('Batch Size: %d\n'       % set_batch_size)
file.write('********************************\n')
file.write('********************************\n')


## Fix random seed for reproducibility
#seed = 42
#numpy.random.seed(seed)


## Set the time 
#now = time.strftime("%X")


# Load UCI data
dataset = numpy.load(data_directory + training_data_sample)                     #Load data from numpy array
testset = numpy.load(data_directory + test_data_sample)                      	#Load data from numpy array


# Split into input (X) and output (Y) variables
X_train_prescale = dataset[:,1:]
Y_train = dataset[:,0]
X_test_prescale = testset[:,1:]
Y_test = testset[:,0]


# Scale
X_train = maxabs_scale(X_train_prescale, axis=0, copy=True)
X_test = maxabs_scale(X_test_prescale, axis=0, copy=True)


# Pull the input layer dimension
input_scale = X_train.shape[1]


for x in range(1, number_of_loops+1):

	file.write('Starting loop %02d\n' %x)

	loop_start_time = time.time()

	
	# Create model
#	model = Sequential()
#
#
#	# Add layers to the model
#	create_model(model, input_scale)
#
#
#	# Compile model
#	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	model = create_model(input_scale)

	# Draw Model
	plot(model,show_shapes=True, to_file='./saved_models/model_%02d.png' %x)
	

	#Announce start of loop
	print "Starting loop %02d" % x	

	# Fit the model and save the history
	history = model.fit(X_train, Y_train, nb_epoch=number_of_epochs, batch_size=set_batch_size)


	# List all data in history
	print(history.history.keys())


	# Plot the accuracy	
	plot_accuracy(history, x, show_toggle=False, save_toggle=False)	

	# Plot the loss
	plot_loss(history, x, show_toggle=False, save_toggle=False)


	# Evaluate the model
	scores = model.evaluate(X_train, Y_train)
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	print "\n"

	# Save the model
	model.save('./saved_models/saved_27D_model_not1000_truemass_%02d.h5' % x)


	# Make predictions
	model_predictions = model.predict(X_test)
	model_class_predictions = model.predict_classes(X_test)


	# Save predictions
	outfile_predict = open('./predictions/model_predictions_not1000.txt', 'w')
	outfile_class_predict = open('./predictions/model_class_predictions_not1000.txt', 'w')
	numpy.savetxt(outfile_predict, model_predictions)
	numpy.savetxt(outfile_class_predict, model_class_predictions)


        # Plot predictions
        plot_predictions(feature, X_test, model_class_predictions, x, show_toggle=False, save_toggle=True)


        # Plot ROC curve
        plot_ROC(Y_test, model_predictions, x, show_toggle=False, save_toggle=True)	

	# Print classification report
        oldStdout = sys.stdout
        #file = open('logFile_training_%02d.txt' % x, 'w')
        sys.stdout = file
	print(classification_report(Y_test, model_class_predictions))
	sys.stdout = oldStdout

	# Reset Model
	model.reset_states() 
	print "Model Reset\n"

	# Log the loop
	loop_end_time = time.time()
	loop_elapsed_time = loop_end_time-loop_start_time
	file.write('Elapsed time for loop %02d: %02d\n\n' %(x, loop_elapsed_time))


overall_end_time = time.time()
overall_elapsed_time = overall_end_time-overall_start_time

print "Done"
print "Elapsed Time = %d seconds" % overall_elapsed_time
file.write('Elapsed time for all %02d loops: %02d\n' %(x, overall_elapsed_time))
file.write('X Feature Count: %d\n'                   % input_scale) 
file.close()
