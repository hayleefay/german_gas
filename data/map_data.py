import pandas as pd

# get all of the individual lat lons and write out to CSV

dataset = pd.read_csv('/gasoline.csv', parse_dates=['date'],
                   index_col=0)

dataset.dropna(inplace=True)
dataset = dataset[['longitude', 'latitude']]
dataset = dataset.drop_duplicates()

dataset.to_csv('latlon.csv')
