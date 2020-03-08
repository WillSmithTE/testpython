import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_x = train_df[['name']]
print(train_x)
train_y = train_df[['Goat-status']]
test_x = test_df[['name']]

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(train_x, train_y)
prediction = neigh.predict(test_x)

print(prediction)
