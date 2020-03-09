import pandas as pd
import numpy as np
import pickle
import json
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from flask import Flask, request, redirect, url_for, flash, jsonify
from flask_cors import CORS

input = 'patty mills'

MODEL_FILE_NAME = 'model.pickle';

train_df = pd.read_csv('train.csv')

def formatPrediction(prediction):
	if prediction == '1':
		return True
	else:
		return False

train_x = train_df[['name']]
train_y = train_df[['Goat-status']]

encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(train_x)
train_x_featurized = encoder.transform(train_x)

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(train_x_featurized, train_y.values.ravel())

pickle.dump(neigh, open(MODEL_FILE_NAME, 'wb'))

app = Flask(__name__)
CORS(app)

@app.route('/goat/<name>', methods=['GET'])
def getPrediction(name):
	data = pd.DataFrame({ 'name': [name] })
	encoded = encoder.transform(data)
	prediction = np.array2string(model.predict(encoded)[0])
	return json.dumps(formatPrediction(prediction))

if __name__ == '__main__':
	model = pickle.load(open(MODEL_FILE_NAME, 'rb'))
	app.run(debug=True, host='0.0.0.0', port=os.environ.get('PORT', '5000'))
