# Author: Michal Lukac, cospelthetraceur@gmail.com
# script for training RNN for rossmann
# You need to have pandas, numpy, scipy and keras

from keras.layers import containers
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, AutoEncoder
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD, Adagrad, Adadelta, RMSprop, Adam

from pandas import HDFStore
import cPickle
import pandas as pd
import numpy as np
import random

from helper import *

# which columns we will use for model
columns = ['CompetitionDistance', 'Promo2', 'Open', 'Promo',
           'StateHoliday_a','StateHoliday_b', 'StateHoliday_c','StateHoliday_0',
           'Assortment_a','Assortment_b','Assortment_c','Assortment_nan',
           'StoreType_a','StoreType_b','StoreType_c','StoreType_d','StoreType_nan',
           'DayOfWeek_1.0','DayOfWeek_2.0','DayOfWeek_3.0','DayOfWeek_4.0','DayOfWeek_5.0','DayOfWeek_6.0','DayOfWeek_7.0',
           'WeekOfMonth_1.0','WeekOfMonth_2.0','WeekOfMonth_3.0','WeekOfMonth_4.0','WeekOfMonth_5.0','WeekOfMonth_6.0',
           'Month_1.0','Month_2.0','Month_3.0','Month_4.0','Month_5.0','Month_6.0','Month_7.0','Month_8.0','Month_9.0','Month_10.0','Month_11.0','Month_12.0',
           'SchoolHoliday','Year_1.0','Year_2.0','Year_3.0','MeanSales', 'MeanVisits', 'MeanDayOfWeekSales1', 'MeanDayOfWeekSales2', 'MeanDayOfWeekSales3', 'MeanDayOfWeekSales4', 'MeanDayOfWeekSales5', 'MeanDayOfWeekSales6', 'MeanDayOfWeekSales7', 'MeanMonthSales1', 'MeanMonthSales10', 'MeanMonthSales11', 'MeanMonthSales12', 'MeanMonthSales2', 'MeanMonthSales3', 'MeanMonthSales4', 'MeanMonthSales5', 'MeanMonthSales6', 'MeanMonthSales7', 'MeanMonthSales8', 'MeanMonthSales9']

print('Loading data ...')
data_dir = '../../data/'
hdf = HDFStore(data_dir + 'data.h5')
data_train = hdf['data_train']
data_test = hdf['data_test']
data_test.sort_index(by=['Id'],ascending=[True])
(DataTr, DataTe) = train_test_split(data_train,0.00)

print('Number of input neurons...', len(columns))
print ('Getting data ...')
X, Y = get_training_dataset_simple(DataTr,columns)
Xtest = get_test_dataset_simple(data_test,columns)

in_neurons = len(columns)
hidden_neurons = 700
hidden_neurons_2 = 250
hidden_neurons_3 = 75
out_neurons = 1
nb_epoch = 10
evaluation = []

# lets create autoencoder
encoder = containers.Sequential([Dense(in_neurons, hidden_neurons, activation='tanh')])
decoder = containers.Sequential([Dense(hidden_neurons, in_neurons, activation='tanh')])

ae = Sequential()
ae.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=False))
ae.compile(loss='mean_squared_error', optimizer='rmsprop')
ae.fit(X, X, verbose=1, nb_epoch = 4)

# lets create second autoencoder
X2 = ae.predict(X)
encoder2 = containers.Sequential([Dense(hidden_neurons, hidden_neurons_2, activation='tanh')])
decoder2 = containers.Sequential([Dense(hidden_neurons_2, hidden_neurons, activation='tanh')])
ae2 = Sequential()
ae2.add(AutoEncoder(encoder=encoder2, decoder=decoder2, output_reconstruction=False))
ae2.compile(loss='mean_squared_error', optimizer='rmsprop')
ae2.fit(X2,X2, verbose=1, nb_epoch = 4)

# lets create third autoencoder
X3 = ae2.predict(X2)
encoder3 = containers.Sequential([Dense(hidden_neurons_2, hidden_neurons_3, activation='tanh')])
decoder3 = containers.Sequential([Dense(hidden_neurons_3, hidden_neurons_2, activation='tanh')])
ae3 = Sequential()
ae3.add(AutoEncoder(encoder=encoder3, decoder=decoder3, output_reconstruction=False))
ae3.compile(loss='mean_squared_error', optimizer='rmsprop')
ae3.fit(X3,X3, verbose=1, nb_epoch = 4)

model = Sequential()
model.add(encoder)
model.add(encoder2)
model.add(encoder3)

print ('Creating simple NN ...')
#model = Sequential()
#model.add(Dense(in_neurons, hidden_neurons, init='uniform', activation='tanh'))
#model.add(Dropout(0.3))
#model.add(Dense(hidden_neurons, hidden_neurons_2, init='uniform',activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(hidden_neurons_3, out_neurons, init='uniform'))
model.compile(loss='mean_squared_error', optimizer='rmsprop')

print ('Getting data ...')
X_train, Y_train = get_training_dataset_simple(DataTr,columns)
X_test = get_test_dataset_simple(data_test,columns)

print ('Fitting model ...')
for k in range(1):
    print(k)
    model.fit(X_train, Y_train, validation_split=0.05, batch_size=15,shuffle=True,nb_epoch=nb_epoch,verbose=1)
    print model.predict(X_train)

print ('Evaluating test ...')
X_train, Y_train = None, None
predicted_values = model.predict(X_test)
data_result = pd.DataFrame({'Sales': predicted_values.astype(int).flatten(), 'Id': data_test['Id'].tolist()})
store_results(data_result, 'test_output.csv')
