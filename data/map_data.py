import pandas as pd
import reverse_geocoder as rg

dataset = pd.read_csv('prepared_gasoline.csv')

dataset = dataset[['latitude', 'longitude']]
dataset = dataset.drop_duplicates()
dataset.dropna(inplace=True)
dataset.reset_index(drop=True, inplace=True)

state_list = []
region_list = []

for index, row in dataset.iterrows():
    result = rg.search((row['latitude'], row['longitude']))
    state = result[0]['admin1']
    region = result[0]['admin2']

    state_list.append(state)
    region_list.append(region)

    if index % 100 == 0:
        print(index)

dataset['state'] = state_list
dataset['region'] = region_list

# add integers for stations
dataset['station'] = pd.Categorical(dataset['longitude'].astype(str)
                                    + dataset['latitude'].astype(str)).codes

dataset.to_csv('latlongeo.csv')
