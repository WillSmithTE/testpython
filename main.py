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

MODEL_FILE_NAME = 'model.pickle';

dataset = pd.read_csv('train.csv')

CATEGORICAL_FEATURES = [ 'college' ]

def formatPrediction(prediction):
	if prediction == '1':
		return True
	else:
		return False

train_y = dataset[['score']]

def transformData(data):
	output = data
	for feature in CATEGORICAL_FEATURES:
		categorical = pd.Categorical(output[feature])
		dummies = pd.get_dummies(categorical, prefix=feature, columns=categorical.unique())
		output = pd.concat([ output, dummies ])
		del output[feature]
	return output

# dataset = transformData(dataset)
del dataset['name']
dataset = pd.get_dummies(dataset)
logging.error(dataset.head().to_string())

del dataset['score']
# del dataset['name']

logging.error(dataset.head().to_string())

classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)
classifier.fit(dataset, train_y)

pickle.dump(classifier, open(MODEL_FILE_NAME, 'wb'))

DERPINPUT = { 'college': ['floridastateuniversity'], 'rings': [0] }
asDataFrame = pd.DataFrame(DERPINPUT)
featurized = transformData(asDataFrame)
derpprediction = classifier.predict(featurized)

logging.error(derpprediction)

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
