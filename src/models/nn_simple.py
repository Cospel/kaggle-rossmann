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

from helper import *

print('Loading data ...')
data_dir = '../../data/'
hdf = HDFStore(data_dir + 'data.h5')
data_train = hdf['data_train']
data_test = hdf['data_test']
data_test.sort_index(by=['Id'],ascending=[True])
(DataTr, DataTe) = train_test_split(data_train,0.01)

in_neurons = 9
hidden_neurons = 450
out_neurons = 1
nb_epoch = 10
evaluation = []

print ('Creating simple NN ...')
model = Sequential()
model.add(Dense(in_neurons, hidden_neurons, init='uniform', activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(hidden_neurons, out_neurons, init='uniform'))
model.compile(loss='mean_squared_error', optimizer='rmsprop')

print ('Getting data ...')
X_train, Y_train = get_training_dataset_simple(DataTr)

print ('Fitting model ...')
for k in range(3):
    print(k)
    model.fit(X_train, Y_train, validation_split=0.0, batch_size=15,shuffle=True,nb_epoch=nb_epoch,verbose=2)
    print model.predict(X_train)

print ('Evaluating test ...')
X_train, Y_train = None, None
X_test = get_test_dataset_simple(data_test)
predicted_values = model.predict(X_test)
data_result = pd.DataFrame({'Sales': predicted_values.flatten(), 'Id': data_test['Id'].tolist()})
store_results(data_result, 'test_output.csv')
