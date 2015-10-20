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

# which columns we will use for model
columns = ['Store', 'CompetitionDistance', 'DayOfWeek', 'Promo2', 'Open',
'Promo', 'StateHoliday', 'SchoolHoliday', 'Month', 'Assortment', 'StoreType',
'WeekOfMonth','Year','MeanCustomers','MeanSales','MeanDayOfWeekSales1','MeanDayOfWeekSales2',
'MeanDayOfWeekSales3','MeanDayOfWeekSales4','MeanDayOfWeekSales5','MeanDayOfWeekSales6',
'MeanDayOfWeekSales7', 'MeanMonthSales1','MeanMonthSales2','MeanMonthSales3','MeanMonthSales4',
'MeanMonthSales5','MeanMonthSales6','MeanMonthSales7','MeanMonthSales8','MeanMonthSales9',
'MeanMonthSales10','MeanMonthSales11','MeanMonthSales12','MeanSalesNotPromo','MeanSalesPromo',
'MeanHolidaySales0','MeanHolidaySales1','MeanHolidaySales2','MeanHolidaySales3']

print('Loading data ...')
data_dir = '../../data/'
hdf = HDFStore(data_dir + 'data.h5')
data_train = hdf['data_train']
data_test = hdf['data_test']
data_test.sort_index(by=['Id'],ascending=[True])
(DataTr, DataTe) = train_test_split(data_train,0.00)
#print DataTr.columns.values

print ('Getting data ...')
X_train, Y_train = get_training_dataset_simple(DataTr,columns)

print ('Fitting model ...')
model = RandomForestRegressor(max_depth=55,n_estimators=70)
model.fit(X_train,Y_train)

print ('Evaluating test ...')
Y_pred = model.predict(X_train)
print ('Eval loss train rmspe:',RMSPE(Y_train,Y_pred))

print('Now eval test ...')
X_train, Y_train = None, None
X_test = get_test_dataset_simple(data_test,columns)
predicted_values = model.predict(X_test)

# lets remove negative values
predicted_values = [0 if i < 0 else i for i in predicted_values.astype(int).flatten()]

data_result = pd.DataFrame({'Sales': predicted_values, 'Id': data_test['Id'].tolist()})
store_results(data_result, 'test_output.csv')
print ('Done')
