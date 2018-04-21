import pandas as pd
import numpy as np
from datetime import datetime
from math import sqrt
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras

# read in the data
df = pd.read_csv("./../data/supervised_1_1.csv")

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

values = df.values
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
reframed = pd.DataFrame(scaler.fit_transform(values))
# frame as supervised learning
n_seq = 1
reframed.head()

# split into train and test sets
# now order needs to be by day rather than by station -- reorder by num_days
# num_days is 16 and 15 is station
reframed.sort_values(by=[16, 15], inplace=True)
day_30 = reframed[16].unique()[-30]
train = reframed.loc[reframed[16] < day_30]
test = reframed.loc[reframed[16] >= day_30]

n_seq = 1
train_values = train.values
test_values = test.values
# split into input and outputs
train_X, train_y = train_values[:, n_seq:], train_values[:, :n_seq]
test_X, test_y = test_values[:, n_seq:], test_values[:, :n_seq]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

neurons = 50
epochs = 500
bs = 1000

print("neurons:", neurons)
print("epochs:", epochs)
print("batch size:", bs)

# design network
model = Sequential()
model.add(LSTM(neurons, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(n_seq))
model.compile(loss='mae', optimizer='adam')
# early stopping
es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                   patience=5, verbose=2, mode='auto')
# fit network
history = model.fit(train_X, train_y, epochs=epochs, batch_size=bs,
                    validation_data=(test_X, test_y), verbose=2, shuffle=False,
                    callbacks=[es])

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 0:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 0:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
print('\n\n')
