from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

regr_rf = RandomForestRegressor(max_depth=30, random_state=2)
regr_rf.fit(train_X, train_y)
