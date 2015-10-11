import cPickle
import pandas as pd
import numpy as np
import random

columns = ['Store', 'DayOfWeek', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday', 'Month', 'Assortment', 'StoreType', 'WeekOfMonth','Year']

def get_training_dataset_simple(datam):
    """
    @param datam: pandas dataframe
    @param n_prev: array of length of sequences
    @return d
    """
    docX = datam.as_matrix(columns=columns)
    docY = datam['Sales'].tolist()
    return docX, np.array(docY)

def get_data_sequence(datam, n_prev=10):
    """
    @param datam : pandas dataframe
    @param n_prev: number of previous examples/timesteps for RNN
    @return: (X,Y) divided data (lists of numpy array)
    """
    docX, docY = [], []
    lendata = len(datam)
    for i in xrange(lendata-n_prev):
        docX.append(datam[columns].iloc[i:i+n_prev].as_matrix())
        docY.append(datam[['Sales']].iloc[i+n_prev])
    return np.array(docX), np.array(docY)

def get_test_dataset_simple(datam):
    """
    @param datam: pandas dataframe
    @param n_prev: array of length of sequences
    @return d
    """
    docX = datam.as_matrix(columns=columns)
    return docX

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
