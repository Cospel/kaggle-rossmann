# Author: Michal Lukac, cospelthetraceur@gmail.com
# script for training RNN for rossmann
# You need to have pandas, numpy, scipy and keras

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD, Adagrad, Adadelta, RMSprop, Adam

from pandas import HDFStore
import cPickle
import pandas as pd
import numpy as np
import random
from random import shuffle
import os.path

from helper import *

print ('Getting data ...')
x = np.concatenate(np.load('bigx.npy'))
y = np.concatenate(np.load('bigy.npy'))

print x.shape[2]
in_neurons = x.shape[2]
hidden_neurons = 500
hidden_neurons_2 = 500
out_neurons = 1
nb_epoch = 10
evaluation = []

print ('Creating simple DLSTM ...')
model = Sequential()
model.add(LSTM(in_neurons, hidden_neurons, return_sequences=False))
#model.add(Dropout(0.3))
#model.add(LSTM(hidden_neurons, hidden_neurons_2, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(hidden_neurons, out_neurons))
model.compile(loss='mean_squared_error', optimizer='rmsprop')

print ('Fitting model ...')
print model.evaluate(x,y,verbose=0)
model.fit(x, y, validation_split=0.05, batch_size=50,shuffle=True,nb_epoch=10,verbose=2)

yk = model.predict(x).flatten()
y = y.flatten()
print RMSPE(y,yk)
print ('---------')

# TODO Evaluation of TEST
# ...

print ('Done ...')
