import pandas as pd
import numpy as np
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from numpy import concatenate
from sklearn.metrics import mean_squared_error
from pprint import pprint
import random

# read in the data
df = pd.read_csv("./../data/changes.csv")

print(df.shape)
df.set_index('date', inplace=True)
df.head()

# try removing the averages so as to not include endogenous variables on RHS
df.drop(['eurusd', 'vehicles'], axis=1, inplace=True)
print(df.shape)
df.head()

# replace the oil prices for the last 30 days with the predictions
oil = pd.read_csv('./../data/linear_oil_yhat.csv')

last_30 = ['2015-11-11', '2015-11-13', '2015-11-14', '2015-11-15',
           '2015-11-16', '2015-11-17', '2015-11-18', '2015-11-19',
           '2015-11-20', '2015-11-21', '2015-11-22', '2015-11-23',
           '2015-11-24', '2015-11-25', '2015-11-26', '2015-11-27',
           '2015-11-28', '2015-11-29', '2015-11-30', '2015-12-01',
           '2015-12-02', '2015-12-03', '2015-12-04', '2015-12-05',
           '2015-12-06', '2015-12-07', '2015-12-08', '2015-12-09',
           '2015-12-10', '2015-12-10']

for index, date in enumerate(last_30):
    df.loc[date, 'rotterdam'] = oil['rot_yhat'][index]
    df.loc[date, 'brent'] = oil['brent_yhat'][index]
    df.loc[date, 'wti'] = oil['wti_yhat'][index]

# now order needs to be by day rather than by station -- reorder by num_days
df.sort_values(by=['num_days', 'station'], inplace=True)
df.head()

# split into train and test sets
day_30 = df['num_days'].unique()[-30]
train = df.loc[df['num_days'] < day_30]
test = df.loc[df['num_days'] >= day_30]

# split into input and outputs
train_X, train_y = train[train.columns.difference(['e5gas', 'station'])], train['e5gas']
test_X, test_y = test[test.columns.difference(['e5gas', 'station'])], test['e5gas']
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

md = 25
ne = 50
mf = 'auto'
mss = 2
msl = 1

print("max_depth:", md)
print("n_estimators:", ne)
print("max_features:", mf)
print("min_samples_split:", mss)
print("min_samples_leaf:", msl)

# fit random forest
model = RandomForestRegressor(max_depth=md, random_state=0,
                              n_estimators=ne, max_features=mf,
                              min_samples_split=mss,
                              min_samples_leaf=msl,
                              n_jobs=-1)
model.fit(train_X, train_y.ravel())
# make a prediction
yhat = model.predict(test_X)

# calculate RMSE
rmse = sqrt(mean_squared_error(test_y, yhat))
print('\nTest RMSE: %.4f' % rmse)
print('\n\n')

test_X['y'] = test_y
test_X['y_hat'] = y_hat

test_X.to_csv("rf_predictions.csv")
