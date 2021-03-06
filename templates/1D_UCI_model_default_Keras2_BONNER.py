import os
import numpy
import pandas
import time
import sys 					#to write out stuff printed to screen
import pickle
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pydot
import graphviz


from keras.models import Sequential
from keras.layers import Dense, Dropout
#from keras.utils.visualize_util import plot
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


# Bring in and set global variables must be done before macros!
import globals
globals.init()

import macros_AWS
from macros_AWS import *


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
data_directory = '/home/rice/jrorie/data/'
#data_directory = '/storage1/users/jtr6/UCI_paper_data_sample/'
training_data_sample = 'not1000_train.npy'
test_data_sample = 'not1000_test.npy'
scaler = 'maxabs'							#Options: 'none', 'maxabs', 'robust_scale' 
feature = 27
number_of_loops = 1							#Total number of loops, is incremented later for functions who's index start at 0
number_of_epochs = 1							#Just what it says, number of epochs never re-indexed
set_batch_size = 1000							#Select batch size

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
file.write('Normalizer : %s\n'	    % scaler)
file.write('Seed value: %d\n'       % seed)
file.write('Feature number: %d\n'   % feature)
file.write('Number of loops: %d\n'  % number_of_loops)
file.write('Number of epochs: %d\n' % number_of_epochs)
file.write('Batch Size: %d\n'       % set_batch_size)
file.write('********************************\n')
file.write('********************************\n')



# Load UCI data
dataset = numpy.load(data_directory + training_data_sample)                     #Load data from numpy array
testset = numpy.load(data_directory + test_data_sample)                      	#Load data from numpy array


# Split into input (X) and output (Y) variables
X_train_prescale = dataset[:,[feature,28]]
Y_train = dataset[:,0]
X_test_prescale = testset[:,[feature,28]]
Y_test = testset[:,0]


# Scale
X_train, X_test = scale_x(X_train_prescale, X_test_prescale, scaler)

# Pull the input layer dimension
if X_train.ndim == 1:
	globals.input_scale = 1
if X_train.ndim == 2:
	globals.input_scale = X_train.shape[1]


for x in range(1, number_of_loops+1):

	file.write('Starting loop %02d\n' %x)

	loop_start_time = time.time()

	#Create the model: all layer addition, compilation, etc	
	model = create_model()

	## Draw Model
	#plot(model,show_shapes=True, to_file='./saved_models/model_%02d.png' %x)

	

	#Announce start of loop
	print "Starting loop %02d" % x	

	# Fit the model and save the history
	#history = model.train_on_batch(X_train, Y_train, nb_epoch=number_of_epochs, batch_size=set_batch_size)
	#(loss, TOB_accuracy) = model.train_on_batch(X_train, Y_train)
	print "Fitting and making history"
	history = model.fit(X_train, Y_train, nb_epoch=number_of_epochs, batch_size=set_batch_size)

	## List all data in history
	#print(history.history.keys())


	## Plot the accuracy	
	#plot_accuracy(history, x, show_toggle=False, save_toggle=False)	

	## Plot the loss
	#plot_loss(history, x, show_toggle=False, save_toggle=False)


#	## Evaluate the model
#	print "Evaluating and scoring"
#	scores = model.evaluate(X_train, Y_train)
#	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#	print "\n"

#	# Save the model
#	model.save('./saved_models/saved_1D_model_not1000_truemass_%02d.h5' % x)


#	# Make predictions
#	model_predictions = model.predict(X_test)
#	model_class_predictions = model.predict_classes(X_test)


#	# Save predictions
#	outfile_predict = open('./predictions/model_predictions_not1000_%02d.txt' % x, 'w')
#	outfile_class_predict = open('./predictions/model_class_predictions_not1000_%02d.txt' %x, 'w')
#	numpy.savetxt(outfile_predict, model_predictions)
#	numpy.savetxt(outfile_class_predict, model_class_predictions)


#       # Plot predictions
#	print "Plotting predictions"
#       plot_predictions_1D(feature, X_test_prescale, model_class_predictions, x, show_toggle=False, save_toggle=True)


#        # Plot ROC curve
#	print "Plotting ROC"
#        plot_ROC(Y_test, model_predictions, x, show_toggle=False, save_toggle=True)	

#	# Print classification report
#        oldStdout = sys.stdout
#        #file = open('logFile_training_%02d.txt' % x, 'w')
#        sys.stdout = file
#	print(classification_report(Y_test, model_class_predictions))
#	sys.stdout = oldStdout
	
	#print "Accuracy = %d" % TOB_accuracy


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
file.write('X Feature Count: %d\n'                   % globals.input_scale) 
file.close()
