import pandas as pd
import numpy as np
#Keras Layers
from keras.layers import Dense, Dropout, Activation
#To build Sequential network
from keras.models import Sequential

from keras.preprocessing.text import Tokenizer
# using reuters data set from keras datasets
from keras.datasets import reuters
import matplotlib.pyplot as plt

# to create callbacks list
from keras.callbacks import EarlyStopping, ModelCheckpoint
# To set pickle = True
old = np.load
np.load = lambda *a,**k: old(*a, allow_pickle=True, **k)


n = 5000
#Loading the data for training and testing
(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words = n)

#Tokenizing
tokenizer = Tokenizer(num_words = n)
X_train_ = tokenizer.sequences_to_matrix(X_train, mode = 'binary')
X_test_ = tokenizer.sequences_to_matrix(X_test, mode = 'binary')


#building the Network

model = Sequential()
model.add(Dense(128, activation = 'relu', input_shape = (n,)))
#Using dropout to handle overfitting if the model
#This creates an ensemble network
model.add(Dropout(0.2))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(46, activation = 'softmax'))

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

#hdf5 files store the info for every epoch, if the Early stopping is occuered, creation of hdf5 files will also be stopped.
filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"

# hdf5 files will save only the best under selected monitor = 'val_acc'
checkpoint = ModelCheckpoint(filepath, monitor = 'val_acc', verbose = 2, save_best_only = True, mode = 'max')

#If the 'val_acc' doesn't improve after 2 epochs, trainig will be stopped after that epoch

#The callback list is passed into the fit function.
callbacks_list = [EarlyStopping(monitor='val_loss', patience=2),checkpoint]


model.fit(X_train_,y_train, epochs = 20, verbose = 2, batch_size = 50, validation_data = (X_test_, y_test), callbacks = callbacks_list)
score = model.evaluate(X_test_, y_test, verbose = 2)
print(score)
