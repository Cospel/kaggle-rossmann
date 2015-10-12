# Author: Michal Lukac, cospelthetraceur@gmail.com
# script for training random forest for rossmann kaggle competition
# You need to have pandas, numpy, scipy and keras

from sklearn.ensemble import RandomForestRegressor

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
(DataTr, DataTe) = train_test_split(data_train,0.00)

print ('Getting data ...')
X_train, Y_train = get_training_dataset_simple(DataTr)

print ('Fitting model ...')
model = RandomForestRegressor(max_depth=40,n_estimators=50)
model.fit(X_train,Y_train)

print ('Evaluating test ...')
Y_pred = model.predict(X_train)
print ('Eval loss train rmspe:',RMSPE(Y_train,Y_pred))

print('Now eval test ...')
X_train, Y_train = None, None
X_test = get_test_dataset_simple(data_test)
predicted_values = model.predict(X_test)
data_result = pd.DataFrame({'Sales': predicted_values.astype(int).flatten(), 'Id': data_test['Id'].tolist()})
store_results(data_result, 'test_output.csv')
print ('Done')
