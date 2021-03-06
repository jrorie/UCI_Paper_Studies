from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima toy mc dataset
dataset = numpy.loadtxt("gauss_and_uniform.txt", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:1]
Y = dataset[:,1]
# create model
model = Sequential()
model.add(Dense(3, input_dim=1, init='uniform', activation='relu'))
#model.add(Dense(1, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, nb_epoch=1, batch_size=10)
# calculate predictions
predictions = model.predict(X)
# round predictions
rounded = [round(x) for x in predictions]
print(rounded)
