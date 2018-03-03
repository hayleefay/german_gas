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
# # define the number of kilometers in one radian
# kms_per_radian = 6371.0088
#
# # scatterplot it to get a sense of what it looks like
# df = sp.sort_values(by=['latitude', 'longitude'])
# # ax = df.plot(kind='scatter', x='longitude', y='latitude', alpha=0.5, linewidth=0)
#
# # represent points consistently as (lat, lon)
# coords = df.as_matrix(columns=['latitude', 'longitude'])
#
# # define epsilon as 10 kilometers, converted to radians for use by haversine
# epsilon = 10 / kms_per_radian
#
# start_time = time.time()
# db = DBSCAN(eps=epsilon, min_samples=10, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
# cluster_labels = db.labels_
# unique_labels = set(cluster_labels)
#
# # get the number of clusters
# num_clusters = len(set(cluster_labels))
#
# # get colors and plot all the points, color-coded by cluster (or gray if not in any cluster, aka noise)
# fig, ax = plt.subplots()
# colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
#
# # for each cluster label and color, plot the cluster's points
# for cluster_label, color in zip(unique_labels, colors):
#
#     size = 150
#     if cluster_label == -1:  #make the noise (which is labeled -1) appear as smaller gray points
#         color = 'gray'
#         size = 30
#
#     # plot the points that match the current cluster label
#     x_coords = coords[cluster_labels == cluster_label][:, 1]
#     y_coords = coords[cluster_labels == cluster_label][:, 0]
#     ax.scatter(x=x_coords, y=y_coords, c=color, edgecolor='k', s=size, alpha=0.5)
#
# ax.set_title('Number of clusters: {}'.format(num_clusters))
# plt.show()

# import shapefile
# shape = shapefile.Reader("./../data/plz-gebiete.shp/plz-gebiete.shp")
# #first feature of the shapefile
# feature = shape.shapeRecords()[0]
# first = feature.shape.__geo_interface__
# print(first)
# {'type': 'LineString', 'coordinates': ((0.0, 0.0), (25.0, 10.0), (50.0, 50.0))}

import pandas as pd
import shapefile
from shapely.geometry import Point # Point class
from shapely.geometry import shape # shape() is a function to convert geo objects through the interface

dataset = pd.read_csv('./../data/gasoline.csv', parse_dates=['date'],
                   index_col=0)

dataset.dropna(inplace=True)
dataset = dataset[['longitude', 'latitude']]

shp = shapefile.Reader('./../data/plz-gebiete.shp/plz-gebiete.shp') #open the shapefile
all_shapes = shp.shapes() # get all the polygons
all_records = shp.records()

for row in dataset:
    point = (row['longitude'], row['latitude'])

    for i in len(all_shapes):
        boundary = all_shapes[i] # get a boundary polygon
        if Point(pt).within(shape(boundary)): # make a point and see if it's in the polygon
           name = all_records[i][2] # get the second field of the corresponding record
           print("The point is in", name)
