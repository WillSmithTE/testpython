import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_x = train_df[['name']]
train_y = train_df[['Goat-status']]
test_x = pd.read_csv('test.csv')[['name']]

encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(train_x)
train_x_featurized = encoder.transform(train_x)
test_x_featurized = encoder.transform(test_x)

neigh = RadiusNeighborsRegressor(radius=1.0)
neigh.fit(train_x_featurized, train_y.values.ravel())
prediction = neigh.predict(test_x_featurized)

if prediction:
	print('GOAT!')
else:
	print('not the goat ...')
