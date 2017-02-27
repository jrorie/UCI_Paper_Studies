from keras.models import Sequential
from keras.layers import Dense
import numpy
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


# Fix random seed for reproducibility
seed = 42
numpy.random.seed(seed)

# Load UCI data

## truncated data set
#dataset = numpy.loadtxt("/storage1/users/jtr6/UCI_paper_data_sample/mods/1000_train.10K.csv", delimiter=",")
#testset = numpy.loadtxt("/storage1/users/jtr6/UCI_paper_data_sample/mods/1000_test.truncated.csv", delimiter=",")
## full data set
dataset = numpy.loadtxt("/storage1/users/jtr6/UCI_paper_data_sample/1000_train.csv", delimiter=",")
testset = numpy.loadtxt("/storage1/users/jtr6/UCI_paper_data_sample/1000_test.csv", delimiter=",")


# Select feature
feature = 27


# Split into input (X) and output (Y) variables
#X = dataset[:,0:1]
#X=dataset[:,27] #Split around x=0 in truth table
X=dataset[:,feature]
Y=dataset[:,0]
X_test = testset[:,feature]


# Create model
model = Sequential()
model.add(Dense(12, input_dim=1, init='uniform', activation='relu')) #Input layer
model.add(Dense(8, init='uniform', activation='relu'))               #Hidden Layer 1
model.add(Dense(10, init='uniform', activation='relu'))              #Hidden Layer 2
model.add(Dense(8, init='uniform', activation='relu'))               #Hidden Layer 3
model.add(Dense(1, init='uniform', activation='sigmoid'))


# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Fit the model
model.fit(X, Y, nb_epoch=5, batch_size=10000)


# Evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# Save the model
model.save('./saved_1D_model.h5')


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
plt.hist(X_test[numpy.nonzero(model_class_predictions.T)[1]], bins_array, color='g')
plt.hist(X_test[numpy.nonzero(model_class_predictions.T==0)[1]], bins_array, color='b')
plt.show()
