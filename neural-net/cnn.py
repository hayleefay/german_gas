import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# set random seed
seed = 7
np.random.seed(seed)

# import data
print("💾 Start importing data")
df = pd.read_csv("./../data/gasoline.csv")
print("✅ Done importing data")

# remove nans
df = df.dropna(axis=0, how='any')

# define features and label
print("✂️ Divide data")
# removing `marke` and `date` for now, figure out how to include string
X = pd.DataFrame.as_matrix(df, columns=['autobahn', 'aral', 'esso',
                                        'jet', 'shell', 'total',
                                        'rotterdam', 'brent', 'wti',
                                        'eurusd', 'vehicles'])
y = pd.DataFrame.as_matrix(df, columns=['e5gas'])

# import pdb; pdb.set_trace()


def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(11, input_dim=11, kernel_initializer='normal',
                    activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))

    # compile model
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model


# create sklearn regressor
estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=1,
                           batch_size=5000, verbose=1)

# evaluate the model with a k-fold cross validation model
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X, y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

# standardize the dataset
print("\n\nStandardize and go again")
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=1,
                                         batch_size=5000, verbose=1)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, y, cv=kfold)
print("Standardized: %.2f(%.2f) MSE" % (results.mean(), results.std()))


# deeper network topology
def larger_model():
    # create model
    model = Sequential()
    model.add(Dense(11, input_dim=11, kernel_initializer='normal',
                    activation='relu'))
    model.add(Dense(5, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


print("\n\nDeepen and go again")
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=larger_model, epochs=1,
                                         batch_size=5000, verbose=1)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, y, cv=kfold)
print("Larger: %.2f (%.2f) MSE" % (results.mean(), results.std()))


# wider network topology
def wider_model():
    # create model
    model = Sequential()
    model.add(Dense(20, input_dim=11, kernel_initializer='normal',
                    activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


print("\n\nWiden and go again")
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=wider_model, epochs=1,
                                         batch_size=5000, verbose=1)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, y, cv=kfold)
print("Wider: %.2f (%.2f) MSE" % (results.mean(), results.std()))
