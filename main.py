import pandas as pd
import logging

from sklearn.neighbors import KNeighborsClassifier

train_df = pd.read_csv('train.csv', index_col=0)

train_x = train_df[['name']]
train_y = train_df[['Goat-status']]

neigh = KNeighborsClassifier(n_neighbors=3)
neight.fit(train_x, train_y)
prediction = neigh.predict('malik beasley')

print(prediction)
