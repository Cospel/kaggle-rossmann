# Author: Michal Lukac, cospelthetraceur@gmail.com
# script for training RNN for rossmann
# You need to have pandas, numpy, scipy and keras

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD, Adagrad, Adadelta, RMSprop, Adam

import cPickle
import pandas as pd
import numpy as np

def get_training_dataset_all(datam):
    """
    @param datam: pandas dataframe
    @param n_prev: array of length of sequences
    @return d
    """
    docX, docY = [], []

    for timesteps in range(len(datam)-1):
        if random.random() < 0.1:
            tempX, tempY = get_data_sequence(datam, timesteps+1)
            for sequence, label in zip(tempX, tempY):
                docX.append(sequence)
                docY.append(np_utils.to_categorical([label], max_value))
    return np.array(docX), np.array(docY)

def get_training_dataset(datam, n_prev=4):
    """
    @param datam: pandas dataframe
    @param n_prev: array of length of sequences
    @return d
    """
    docX, docY = [], []
    tempX, tempY = get_data_sequence(datam, n_prev=n_prev)
    for sequence, label in zip(tempX, tempY):
        docX.append(sequence)
        docY.append(label)

    # uncoment this if categorical
    #docY = np_utils.to_categorical(docY, max_value)

    if len(docX) == 0:
        return get_training_dataset(datam, n_prev=n_prev-1)
    return np.array(docX), np.array(docY)

def get_data_sequence(datam, n_prev=2):
    """
    @param datam : pandas dataframe
    @param n_prev: number of previous examples/timesteps for RNN
    @return: (X,Y) divided data (lists of numpy array)
    """
    docX, docY = [], []
    for i in range(len(datam)-n_prev):
        docX.append(datam[['place_asked','place_answered','place_answered_binary']].iloc[i:i+n_prev].as_matrix())
        docY.append(datam['place_answered_binary'].iloc[i+n_prev])
        # uncoment this if categorical
        #docY.append(datam['place_asked'].iloc[i+n_prev])
    return (docX, docY)

def train_test_split(df, test_size=0.1):
    """
    @param df: dataframe
    @param test_size: Percentage of how many samples will be in test dataset
    @return: return two dataframes for train and test
    """
    n_examples = int(round(len(df)*(1-test_size)))
    data_train = df.iloc[0:n_examples]
    data_test = df.iloc[n_examples:]
    return (data_train, data_test)

print('Loading data ...')
data_dir = '../../data/'
hdf = HDFStore(data_dir + 'data.h5')
data_train = hdf['data_train']

(DataTr, DataTe) = train_test_split(data_train,0.01)

in_neurons = 30
hidden_neurons = 650
hidden_neurons_2 = 900
hidden_neurons_3 = 1050
out_neurons = 65
nb_epoch = 10
evaluation = []

print ('Creating 2DLSTM ...')
model = Sequential()
model.add(LSTM(in_neurons, hidden_neurons, return_sequences=True))
model.add(LSTM(hidden_neurons, hidden_neurons_2, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(hidden_neurons_2, hidden_neurons_3, return_sequences=False))
model.add(Dropout(0.1))
model.add(Dense(hidden_neurons_3, out_neurons, init='uniform'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', class_mode='categorical')

print ('Fitting model ...', i)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10)
for k in range(nb_epoch):
    print(k)
    model.fit(X_train, Y_train, validation_split=0.0, batch_size=15,shuffle=True,nb_epoch=nb_epoch,verbose=2,show_accuracy=True)
    print ('Evaluate model')
    evaluation.append((model.evaluate(X_test, Y_test, show_accuracy=True), i, k*nb_epoch))
