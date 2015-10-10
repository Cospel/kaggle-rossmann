import cPickle
import pandas as pd
import numpy as np
import random

columns = ['Store', 'DayOfWeek', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday', 'Month', 'Assortment', 'StoreType']

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

def get_training_dataset_sequence(datam, n_prev=1):
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

    return np.array(docX), np.array(docY)

def get_training_dataset_simple(datam):
    """
    @param datam: pandas dataframe
    @param n_prev: array of length of sequences
    @return d
    """
    docX, docY = [], []
    tempX, tempY = get_data_sequence(datam, n_prev=0)
    for sequence, label in zip(tempX, tempY):
        docX.append(sequence)
        docY.append(label)

    return np.array(docX), np.array(docY)

def get_data_sequence(datam, n_prev=1):
    """
    @param datam : pandas dataframe
    @param n_prev: number of previous examples/timesteps for RNN
    @return: (X,Y) divided data (lists of numpy array)
    """
    docX, docY = [], []
    lendata = len(datam)
    if n_prev == 0:
        for i in xrange(lendata-n_prev):
            #print(str(i)+'/'+str(lendata))
            # this is simple data for non rnn models
            docX.append(datam[columns].iloc[i].as_matrix())
            docY.append(datam['Sales'].iloc[i])
    else:
        for i in xrange(lendata-n_prev):
            #print(str(i)+'/'+str(lendata))
            # this are data sequences for rnn models
            docX.append(datam[columns].iloc[i:i+n_prev].as_matrix())
            docY.append(datam['Sales'].iloc[i+n_prev])
    return (docX, docY)

def get_test_data_sequence(datam, n_prev=1):
    """
    @param datam : pandas dataframe
    @param n_prev: number of previous examples/timesteps for RNN
    @return: (X,Y) divided data (lists of numpy array)
    """
    docX = []
    lendata = len(datam)
    if n_prev == 0:
        for i in xrange(lendata-n_prev):
            #print(str(i)+'/'+str(lendata))
            # this is simple data for non rnn models
            docX.append(datam[columns].iloc[i].as_matrix())
    else:
        for i in xrange(lendata-n_prev):
            #print(str(i)+'/'+str(lendata))
            # this are data sequences for rnn models
            docX.append(datam[columns].iloc[i:i+n_prev].as_matrix())
    return docX

def get_test_dataset_simple(datam):
    """
    @param datam: pandas dataframe
    @param n_prev: array of length of sequences
    @return d
    """
    tempX = get_test_data_sequence(datam, n_prev=0)
    return np.array(tempX)

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

def store_results(dataframe, output_file):
    dataframe[[ 'Id', 'Sales' ]].to_csv( output_file, index = False )
