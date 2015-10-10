# Author: Michal Lukac, cospelthetraceur@gmail.com
# script for geting hdf5 dataset for rossmann kaggle competition
# You need to have pandas, numpy

from pandas import HDFStore
import pandas as pd
import numpy as np


def load_data_file(filename,dtypes,parsedate = True):
    date_parse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
    if parsedate:
        return pd.read_csv(filename, sep=',', parse_dates=['Date'], date_parser= date_parse,dtype=dtypes)
    else:
        return pd.read_csv(filename, sep=',', dtype=dtypes)


# Load data, parse data, clean unwanted columns, replace nan values, create column
print('Loading data ...')
data_dir = '../../data/'
data_train = load_data_file(data_dir + 'train.csv',
                            {'Id':np.int32,
                             'Store':np.int32,
                             'DayOfWeek':np.int8,
                             'Sales':np.int32,
                             'Customers':np.int32,
                             'Open':np.int8,
                             'Promo':np.int8,
                             'StateHoliday':np.object, # categorical
                             'SchoolHoliday':np.int8})

data_test = load_data_file(data_dir + 'test.csv',
                            {'Id':np.int32,
                             'Store':np.int32,
                             'DayOfWeek':np.int8,
                             'Open':np.object,         # there is some nan values
                             'Promo':np.object,
                             'StateHoliday':np.object, # categorical
                             'SchoolHoliday':np.int8})

data_store = load_data_file(data_dir + 'store.csv',
                            {'Store':np.int32,
                             'StoreType':np.object,
                             'Assortment':np.object,
                             'CompetitionDistance':np.object,
                             'CompetitionOpenSiceMonth':np.object, # categorical
                             'CompetitionOpenSiceYear':np.object,
                             'Promo2':np.int8,
                             'Promo2SinceWeek':np.object,
                             'Promo2SinceYear':np.object,
                             'PromoInterval':np.object}, False)

hdf = HDFStore(data_dir + 'data.h5')
hdf.put('data_train', data_train, format='table', data_columns=True)
hdf.put('data_test', data_test, format='table', data_columns=True)
hdf.put('data_store', data_store, format='table', data_columns=True)
print('Done ...')
