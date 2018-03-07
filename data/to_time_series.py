import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# read in the data
dataset = pd.read_csv('prepared_gasoline.csv')

# read in the geo data
geo = pd.read_csv('latlongeo.csv')
geo.drop(['Unnamed: 0', 'region'], axis=1, inplace=True)

# join them -- inner drops the lat/lon nan values
dataset = dataset.merge(geo, how='inner', on=['latitude', 'longitude'])

# add global mean of gas price for each observation
global_mean = dataset.groupby('date')['e5gas'].mean()
global_df = global_mean.to_frame()
global_df.rename(columns={"e5gas": "global_mean"}, inplace=True)
dataset = dataset.merge(global_df, right_index=True, left_on='date')

# add regional mean of gas price for each observation
state_mean = df.groupby(['date', 'state'])['e5gas'].mean()
state_df = state_mean.to_frame()
state_df.rename(columns={"e5gas": "state_mean"}, inplace=True)
dataset = dataset.merge(state_df, right_index=True, left_on=['date', 'state'])

# drop column with state and lat/lon string
dataset.drop(['latitude', 'longitude'], axis=1, inplace=True)

# one hot encode state and marke
dataset['state'] = pd.Categorical(dataset['state']).codes

# 12724 of the stations have 575 observations

# for now, choose one station and then write it out to CSV and we will fit the
# neural net!

# divide data so that there is a separate df for each station

# sort by index
# .sort_index()

# within each station df, add n lags to each observation and write to files
# start with just one lag -- so just move everything down one observation and then
# drop any with NAs

# create a script that loops through the files and builds a random forest for each one
# and for each time period wanting to be predicted (RF only predicts the next point,
# look at the Kaggle RF Matlab code to see how he preps the data for future points)

# we can basically follow the Keras LSTM tutorial at this point also

# import necessary modules
# import pandas as pd, numpy as np, matplotlib.pyplot as plt, time
# from sklearn.cluster import DBSCAN
# from sklearn import metrics
# from geopy.distance import great_circle
# from shapely.geometry import MultiPoint
# from pandas import read_csv
# from datetime import datetime
#
# def parse(x):
#     return datetime.strptime(x, '%d%b%Y').strftime('%Y %m %d')
#
#
# dataset = read_csv('./../data/gasoline.csv', parse_dates=['date'],
#                    index_col=0, date_parser=parse)
#
# # don't drop marke in future
# dataset.drop(['mts_id', 'intid', 'marke', 'year', 'month', 'day', 'vehicles1',
#               'latitudezst', 'longitudezst', 'brentl', 'd1', 'zst1'],
#              axis=1, inplace=True)
# dataset.set_index('date', inplace=True)
#
# global_mean = dataset.groupby('date')['e5gas'].mean()
# df = global_mean.to_frame()
# df.rename(columns={"e5gas": "global_mean"}, inplace=True)
# sp = pd.merge(dataset, df, left_index=True, right_index=True)
#
# sp.keys()
# print(sp.shape)
# sp.dropna(inplace=True)
# print(sp.shape)
#


# # convert series to supervised learning
# def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
#     n_vars = 1 if type(data) is list else data.shape[1]
#     df = DataFrame(data)
#     cols, names = list(), list()
#     # input sequence (t-n, ... t-1)
#     for i in range(n_in, 0, -1):
#         cols.append(df.shift(i))
#         names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
#     # forecast sequence (t, t+1, ... t+n)
#     for i in range(0, n_out):
#         cols.append(df.shift(-i))
#         if i == 0:
#             names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
#         else:
#             names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
#     # put it all together
#     agg = concat(cols, axis=1)
#     agg.columns = names
#     # drop rows with NaN values
#     if dropnan:
#         agg.dropna(inplace=True)
#     return agg
#
#
# # load dataset
# dataset = read_csv('pollution.csv', header=0, index_col=0)
# values = dataset.values
# # ensure all data is float
# values = values.astype('float32')
# # normalize features
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled = scaler.fit_transform(values)
# # frame as supervised learning
# reframed = series_to_supervised(scaled, 1, 1)
# # drop columns we don't want to predict
# reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
# print(reframed.head())
