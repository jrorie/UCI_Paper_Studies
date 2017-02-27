from keras.models import Sequential
from keras.layers import Dense
import numpy
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


# fix random seed for reproducibility
seed = 42
numpy.random.seed(seed)

# load combined toy mc dataset
dataset = numpy.loadtxt("gauss_and_uniform.txt", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:,0:1]
Y = dataset[:,1]

# create model
model = Sequential()
model.add(Dense(12, input_dim=1, init='uniform', activation='relu'))
#model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, nb_epoch=5, batch_size=1000)

# Evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Save the model
model.save('./saved_toy_model.h5')

# Make predictions
model_predictions = model.predict(X)
model_class_predictions = model.predict_classes(X)

# Save predictions
outfile_predict = open('model_predictions.txt', 'w')
outfile_class_predict = open('model_class_predictions.txt', 'w')
numpy.savetxt(outfile_predict, model_predictions)
numpy.savetxt(outfile_class_predict, model_class_predictions)


# Plot predictions
bins_array = numpy.arange(-5,5,0.1)
plt.hist(X[numpy.nonzero(model_class_predictions.T)[1]], bins_array, color='g')
plt.hist(X[numpy.nonzero(model_class_predictions.T==0)[1]], bins_array, color='b')
plt.show()
