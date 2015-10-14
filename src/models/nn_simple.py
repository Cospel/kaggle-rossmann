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

# which columns we will use for model
columns = ['Store', 'CompetitionDistance', 'Promo2', 'Open', 'Promo',
           'StateHoliday_a','StateHoliday_b', 'StateHoliday_c','StateHoliday_0',
           'Assortment_a','Assortment_b','Assortment_c','Assortment_nan',
           'StoreType_a','StoreType_b','StoreType_c','StoreType_d','StoreType_nan',
           'DayOfWeek_1.0','DayOfWeek_2.0','DayOfWeek_3.0','DayOfWeek_4.0','DayOfWeek_5.0','DayOfWeek_6.0','DayOfWeek_7.0',
           'WeekOfMonth_1.0','WeekOfMonth_2.0','WeekOfMonth_3.0','WeekOfMonth_4.0','WeekOfMonth_5.0','WeekOfMonth_6.0',
           'Month_1.0','Month_2.0','Month_3.0','Month_4.0','Month_5.0','Month_6.0','Month_7.0','Month_8.0','Month_9.0','Month_10.0','Month_11.0','Month_12.0',
           'SchoolHoliday']

print('Loading data ...')
data_dir = '../../data/'
hdf = HDFStore(data_dir + 'data.h5')
data_train = hdf['data_train']
data_test = hdf['data_test']
data_test.sort_index(by=['Id'],ascending=[True])
(DataTr, DataTe) = train_test_split(data_train,0.00)

print('Number of input neurons...', len(columns))
in_neurons = len(columns)
hidden_neurons = 400
hidden_neurons_2 = 100
out_neurons = 1
nb_epoch = 10
evaluation = []

print ('Creating simple NN ...')
model = Sequential()
model.add(Dense(in_neurons, hidden_neurons, init='uniform', activation='tanh'))
model.add(Dropout(0.3))
model.add(Dense(hidden_neurons, hidden_neurons_2, init='uniform'))
model.add(Dropout(0.2))
model.add(Dense(hidden_neurons_2, out_neurons, init='uniform'))
model.compile(loss='mean_squared_error', optimizer='rmsprop')

print ('Getting data ...')
X_train, Y_train = get_training_dataset_simple(DataTr,columns)
X_test = get_test_dataset_simple(data_test,columns)

print ('Fitting model ...')
for k in range(3):
    print(k)
    model.fit(X_train, Y_train, validation_split=0.05, batch_size=15,shuffle=True,nb_epoch=nb_epoch,verbose=1)
    print model.predict(X_train)

print ('Evaluating test ...')
X_train, Y_train = None, None
predicted_values = model.predict(X_test)
data_result = pd.DataFrame({'Sales': predicted_values.astype(int).flatten(), 'Id': data_test['Id'].tolist()})
store_results(data_result, 'test_output.csv')
