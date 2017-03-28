from keras.models import Sequential
from keras.layers import Dense
import numpy
from sklearn.metrics import roc_curve, auc
import pandas as pd
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import time 


# Fix random seed for reproducibility
seed = 42
numpy.random.seed(seed)

# Set the time 
now = time.strftime("%X")

# Load UCI data
## truncated data set
#dataset = numpy.loadtxt("/storage1/users/jtr6/UCI_paper_data_sample/mods/not1000_train.10K.csv", delimiter=",")
#testset = numpy.loadtxt("/storage1/users/jtr6/UCI_paper_data_sample/mods/not1000_test.truncated.csv", delimiter=",")
## full data set
#dataset = numpy.loadtxt("/storage1/users/jtr6/UCI_paper_data_sample/not1000_train.csv", delimiter=",")
#testset = numpy.loadtxt("/storage1/users/jtr6/UCI_paper_data_sample/not1000_test.csv", delimiter=",")
#numpy.save('/storage1/users/jtr6/UCI_paper_data_sample/not1000_train.npy', dataset)
#numpy.save('/storage1/users/jtr6/UCI_paper_data_sample/not1000_test.npy', testset)
dataset = numpy.load('/storage1/users/jtr6/UCI_paper_data_sample/not1000_train.npy')                     #Load data from numpy array
testset = numpy.load('/storage1/users/jtr6/UCI_paper_data_sample/not1000_test.npy')                      #Load data from numpy array

# Select feature
feature = 27


# Split into input (X) and output (Y) variables
#X_train = dataset[:,[feature,28]]
X_train = dataset[:,feature]
Y_train = dataset[:,0]
X_test = testset[:,feature]
#X_test = testset[:,[feature,28]]
Y_test = testset[:,0]

# Pull the input layer dimension
#input_scale = X_train.shape[1]

# Create model
model = Sequential()
#model.add(Dense(12, input_dim=input_scale, init='uniform', activation='relu')) #Input layer
model.add(Dense(12, input_dim=1, init='uniform', activation='relu')) #Input layer
model.add(Dense(8, init='uniform', activation='relu'))               #Hidden Layer 1
model.add(Dense(10, init='uniform', activation='relu'))              #Hidden Layer 2
model.add(Dense(8, init='uniform', activation='relu'))               #Hidden Layer 3
model.add(Dense(1, init='uniform', activation='sigmoid'))


# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Fit the model and save the history
history = model.fit(X_train, Y_train, nb_epoch=10, batch_size=10000)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('1D_not1000_notruemass_accuracy_%s.png' % now)
plt.show()
plt.clf()


# summarize history for loss
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('1D_not1000_notruemass_loss_%s.png' % now)
plt.show()
plt.clf()

# Evaluate the model
scores = model.evaluate(X_train, Y_train)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# Save the model
model.save('./saved_1D_model_not1000_notruemass.h5')


# Make predictions
model_predictions = model.predict(X_test)
model_class_predictions = model.predict_classes(X_test)


# Save predictions
outfile_predict = open('model_predictions.txt', 'w')
outfile_class_predict = open('model_class_predictions.txt', 'w')
numpy.savetxt(outfile_predict, model_predictions)
numpy.savetxt(outfile_class_predict, model_class_predictions)


# Setup plot
bins_array = numpy.arange(-5,5,0.1)
plt.xlabel('Feature %d' % feature)
plt.ylabel('Events')
plt.title('Distro of Variable')



# Plot predictions
#plt.hist(X_test[numpy.nonzero(model_class_predictions.T)[1]][:,0], bins_array, color='g')
#plt.hist(X_test[numpy.nonzero(model_class_predictions.T==0)[1]][:,0], bins_array, color='b')
plt.hist(X_test[numpy.nonzero(model_class_predictions.T)[1]], bins_array, color='g')
plt.hist(X_test[numpy.nonzero(model_class_predictions.T==0)[1]], bins_array, color='b')
now = time.strftime("%X")
plt.savefig('1D_not1000_notruemass_predictions_%s.png' % now)
plt.show()

#Plot ROC curve
fpr, tpr, _ = roc_curve(Y_test, model_predictions)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.show()
print('AUC: %f' % roc_auc)
plt.savefig('1D_not1000_notruemass_ROC_%s.png' % now)

