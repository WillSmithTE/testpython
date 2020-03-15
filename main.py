import pandas as pd
import numpy as np
import pickle
import json
import os
import logging

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from flask import Flask, request, redirect, url_for, flash, jsonify
from flask_cors import CORS

MODEL_FILE_NAME = 'model.pickle'

dataset = pd.read_csv('train.csv')

CATEGORICAL_FEATURES = [ 'college' ]

def formatPrediction(prediction):
	if prediction == '100':
		return 'G O A T'
	else:
		return prediction + ' ? Definitely not a goat ...'

train_y = dataset[['score']]

def transformData(data):
	output = data
	for feature in CATEGORICAL_FEATURES:
		categorical = pd.Categorical(output[feature])
		dummies = pd.get_dummies(categorical, prefix=feature, columns=categorical.unique())
		output = pd.concat([ output, dummies ])
		del output[feature]
	return output

del dataset['name']
del dataset['score']
dataset = pd.get_dummies(dataset)


classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), max_iter=2000, random_state=1)
classifier.fit(dataset, train_y.values.ravel())

# pickle.dump(classifier, open(MODEL_FILE_NAME, 'wb'))

app = Flask(__name__)
CORS(app)

@app.route('/goat', methods=['GET'])
def getPrediction():
	college = request.args.get('college')
	rings = request.args.get('rings')
	asDataFrame = pd.DataFrame({ 'college_' + college: [1], 'rings': [rings] })
	asDataFrame = asDataFrame.reindex(columns = dataset.columns, fill_value=0)
	prediction = np.array2string(classifier.predict(asDataFrame)[0])
	return json.dumps(formatPrediction(prediction))

if __name__ == '__main__':
	# model = pickle.load(open(MODEL_FILE_NAME, 'rb'))
	app.run(debug=True, host='0.0.0.0', port=os.environ.get('PORT', '5000'))
