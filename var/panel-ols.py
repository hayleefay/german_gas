import numpy as np
from statsmodels.datasets import grunfeld
from linearmodels import PanelOLS
import pandas as pd

data = pd.read_csv('./../data/prepared_gasoline.csv')
# data = grunfeld.load_pandas().data
# data.year = data.year.astype(np.int64)
# MultiIndex, entity - time
data['station'] = pd.Categorical(data['longitude'].astype(str) + data['latitude'].astype(str)).codes
data = data.set_index(['station', 'date'])

# should probbaly split the data here to leave out the last month for
# prediction a little tricky because it sorts by index now which means
# it is station and then date

import pdb; pdb.set_trace()

# entity_effect? Fixed effects for entity -- do I need to add encoding for station
# station econding removes all intuition about the distance between the stations
# I guess I could leave them as X features
# ValueError: cannot reshape array of size 8452738 into shape (1,575,14699)
mod = PanelOLS(data.e5gas, data[['weekday', 'dautobahn', 'autobahn', 'aral',
                                 'esso', 'jet', 'shell', 'total', 'rotterdam',
                                 'brent', 'wti', 'eurusd', 'vehicles']])
res = mod.fit(cov_type='clustered', cluster_entity=True)
