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

print('Loading data ...')
data_dir = '../../data/'
hdf = HDFStore(data_dir + 'data.h5')
data_train = hdf['data_train']
data_train['Date'] = pd.to_datetime(data_train.Date)
data_train = data_train.ix[pd.to_datetime(data_train.Date).order().index]
(DataTr, DataTe) = train_test_split(data_train,0.00)

print ('Getting data ...')
stores = DataTr['Store'].unique()

big_x = []
big_y = []
i = 0
print ('Fitting model ...')
for epoch in range(1):
    for store in stores:
        i++;
        print (i)
        data = DataTr[DataTr.Store == store]
        x, y = get_data_sequence(data,n_prev=7)
        big_x.apend(x)
        big_y.apend(y)


np.save(big_x,'big_7x.npy')
np.save(big_y,'big_7y.npy')

print ('Done ...')
