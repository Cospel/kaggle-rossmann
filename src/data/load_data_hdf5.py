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

def replace_values(dataframe, column, dictionary):
    return dataframe[column].apply( lambda x: int(dictionary[x]) )

# some known things about dataset
StateHoliday = {'a': 1, 'b': 2, 'c': 3, '0': 0, 0: 0}
Assortment = {'a': 0, 'b': 1, 'c': 2}
StoreType = {'a': 0,'b': 1,'c': 2, 'd': 3}

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

print('Replacing values as ...')
data_train['StateHoliday'] = replace_values(data_train,'StateHoliday', StateHoliday).astype(np.int8)
data_store['Assortment'] = replace_values(data_store,'Assortment', Assortment).astype(np.int8)
data_store['StoreType'] = replace_values(data_store,'StoreType', StoreType).astype(np.int8)
data_test['StateHoliday'] = replace_values(data_test,'StateHoliday', StateHoliday).astype(np.int8)
data_test.loc[data_test.Open.isnull(), 'Open'] = 1
data_test['Open'] = data_test['Open'].astype(np.int8)

print('Add some more features ...')
# we have dayofweek already
data_train['Year'] = data_train['Date'].apply(lambda x: int(str(x)[:4]))
data_train['Month'] = data_train['Date'].apply(lambda x: int(str(x)[5:7]))
data_test['Year'] = data_test['Date'].apply(lambda x: int(str(x)[:4]))
data_test['Month'] = data_test['Date'].apply(lambda x: int(str(x)[5:7]))

print('Create ultimate data')
# this is concatenating datasets including info from stores
data_ut_train = pd.merge(data_train,data_store, on='Store')
data_ut_test  = pd.merge(data_test,data_store, on='Store')

print data_ut_train[0:2]
print ('...')

assert( len( data_ut_train ) == len( data_train ))
assert( len( data_ut_test ) == len( data_test ))

print('Storing data ...')
hdf = HDFStore(data_dir + 'data.h5')
hdf.put('data_train', data_ut_train, format='table', data_columns=True)
hdf.put('data_test', data_ut_test, format='table', data_columns=True)
hdf.put('data_store', data_store, format='table', data_columns=True)

print('Done ...')
