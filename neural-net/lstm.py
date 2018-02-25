'''
Maybe the (n, steps, input_features) can be a tuple for each location in every
time period so it would be input_features = 350 ish. steps seems like it should
be about 10 or so. Not sure what n is.
'''
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM


def _load_data(data, n_prev=1):
    """
    data should be pd.DataFrame()
    """

    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].as_matrix())
        docY.append(data.iloc[i+n_prev].as_matrix())
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY


def train_test_split(df, test_size=0.1):
    """
    This just splits data to training and testing parts
    """
    ntrn = round(len(df) * (1 - test_size))

    X_train, y_train = _load_data(df.iloc[0:ntrn])
    X_test, y_test = _load_data(df.iloc[ntrn:])

    return (X_train, y_train), (X_test, y_test)


print("read in data")
data = pd.read_csv("./../data/grouped_gasoline.csv")

print("build the model")
in_out_neurons = 232304
hidden_neurons = 300

model = Sequential()
model.add(LSTM(hidden_neurons, return_sequences=False,
               input_shape=(None, in_out_neurons)))
model.add(Dense(in_out_neurons, input_dim=hidden_neurons))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error", optimizer="rmsprop")

print("separate train and test")
(X_train, y_train), (X_test, y_test) = train_test_split(data)

# should write these to csv so I don't have to do it every time

print("train the model")
# and now train the model
# batch_size should be appropriate to your memory size
# number of epochs should be higher for real world problems
model.fit(X_train, y_train, batch_size=50, epochs=1, validation_split=0.05)

print("root mean squared error")
predicted = model.predict(X_test)
rmse = np.sqrt(((predicted - y_test) ** 2).mean(axis=0))

# import pandas as pd
# from datetime import datetime
#
#
# def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
#     """
#     Frame a time series as a supervised learning dataset.
#     Arguments:
#         data: Sequence of observations as a list or NumPy array.
#         n_in: Number of lag observations as input (X).
#         n_out: Number of observations as output (y).
#         dropnan: Boolean whether or not to drop rows with NaN values.
#     Returns:
#         Pandas DataFrame of series framed for supervised learning.
#     """
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
#     agg = pd.concat(cols, axis=1)
#     agg.columns = names
#     # drop rows with NaN values
#     if dropnan:
#         agg.dropna(inplace=True)
#     return agg



# # load dataset
# series = read_csv("./../data/gasoline.csv", header=0, parse_dates=['date'],
#                   index_col=0, squeeze=True, date_parser=parser)
#
# # drop columns
# series = series.drop(['mts_id', 'intid', 'date', 'marke'], axis=1)
#
# # sort by date and reindex
# series = series.sort_values(by=['year', 'month', 'day'])
# series = series.reset_index(drop=True)






# from pandas import Series
# from pandas import concat
# from pandas import read_csv
# from pandas import datetime
# from pandas import DataFrame
# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import MinMaxScaler
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from math import sqrt
# from matplotlib import pyplot
# import numpy
#
#
# # load dataset
# def parser(x):
#     return datetime.strptime(x, '%d%b%Y')
#
#
# # frame a sequence as a supervised learning problem
# def timeseries_to_supervised(data, lag=1):
#     df = DataFrame(data)
#     columns = [df.shift(i) for i in range(1, lag+1)]
#     columns.append(df)
#     df = concat(columns, axis=1)
#     df.fillna(0, inplace=True)
#     return df
#
#
# # create a differenced series
# def difference(dataset, interval=1):
#     diff = list()
#     for i in range(interval, len(dataset)):
#         value = dataset[i] - dataset[i - interval]
#         diff.append(value)
#     return Series(diff)
#
#
# # invert differenced value
# def inverse_difference(history, yhat, interval=1):
#     return yhat + history[-interval]
#
#
# # scale train and test data to [-1, 1]
# def scale(train, test):
#     # fit scaler
#     scaler = MinMaxScaler(feature_range=(-1, 1))
#     scaler = scaler.fit(train)
#     # transform train
#     train = train.reshape(train.shape[0], train.shape[1])
#     train_scaled = scaler.transform(train)
#     # transform test
#     test = test.reshape(test.shape[0], test.shape[1])
#     test_scaled = scaler.transform(test)
#     return scaler, train_scaled, test_scaled
#
#
# # inverse scaling for a forecasted value
# def invert_scale(scaler, X, value):
#     new_row = [x for x in X] + [value]
#     array = numpy.array(new_row)
#     array = array.reshape(1, len(array))
#     inverted = scaler.inverse_transform(array)
#     return inverted[0, -1]
#
#
# # fit an LSTM network to training data
# def fit_lstm(train, batch_size, nb_epoch, neurons):
#     X, y = train[:, 0:-1], train[:, -1]
#     X = X.reshape(X.shape[0], 1, X.shape[1])
#     model = Sequential()
#     model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1],
#                    X.shape[2]), stateful=True))
#     model.add(Dense(1))
#     model.compile(loss='mean_squared_error', optimizer='adam')
#     for i in range(nb_epoch):
#         model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0,
#                   shuffle=False)
#         model.reset_states()
#     return model
#
#
# # make a one-step forecast
# def forecast_lstm(model, batch_size, X):
#     X = X.reshape(1, 1, len(X))
#     yhat = model.predict(X, batch_size=batch_size)
#     return yhat[0, 0]
#
#
# # load dataset
# series = read_csv("./../data/gasoline.csv", header=0, parse_dates=['date'],
#                   index_col=0, squeeze=True, date_parser=parser)
#
# # drop columns
# series = series.drop(['mts_id', 'intid', 'date', 'marke'], axis=1)
#
# # sort by date and reindex
# series = series.sort_values(by=['year', 'month', 'day'])
# series = series.reset_index(drop=True)
#
# # summarize first few rows
# print(series.head())
#
# # transform data to be stationary
# raw_values = series.values
# diff_values = difference(raw_values, 1)
#
# # transform data to be supervised learning
# supervised = timeseries_to_supervised(diff_values, 1)
# supervised_values = supervised.values
#
# # Observations from May 16, 2014 to December 11, 2015
#
# # split data into train and test-sets
# train, test = supervised_values[0:7825898], supervised_values[7825898:]
#
# # transform the scale of the data
# scaler, train_scaled, test_scaled = scale(train, test)
#
# # fit the model
# lstm_model = fit_lstm(train_scaled, 1, 3000, 4)
# # forecast the entire training dataset to build up state for forecasting
# train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
# lstm_model.predict(train_reshaped, batch_size=5000)
#
# # walk-forward validation on the test data
# predictions = list()
# for i in range(len(test_scaled)):
#     # make one-step forecast
#     X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
#     yhat = forecast_lstm(lstm_model, 1, X)
#     # invert scaling
#     yhat = invert_scale(scaler, X, yhat)
#     # invert differencing
#     yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
#     # store forecast
#     predictions.append(yhat)
#     expected = raw_values[len(train) + i + 1]
#     print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))
#
# # report performance
# rmse = sqrt(mean_squared_error(raw_values[-12:], predictions))
# print('Test RMSE: %.3f' % rmse)
