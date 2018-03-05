# add a variable for global mean of gas price and regional mean for each observation
# global mean -- try to not cast as dataframe
# global_mean = th.groupby('date')['e5gas'].mean()
# df = global_mean.to_frame()
# df.rename(columns={"e5gas": "global_mean"}, inplace=True)

# regional mean
# found some nan lat/long values in data: 2118

# one hot encode marke, do it separately for each file so that there are not so many brands
# pd.get_dummies(dataset['marke'].str.lower())

# divide data so that there is a separate df for each station

# within each station df, add n lags to each observation and write to files
# def table2lags(table, max_lag, min_lag=0, separator='_'):
#     """ Given a dataframe, return a dataframe with different lags of all its columns """
#     values=[]
#     for i in range(min_lag, max_lag + 1):
#         values.append(table.shift(i).copy())
#         values[-1].columns = [c + separator + str(i) for c in table.columns]
#     return pd.concat(values, axis=1)

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
