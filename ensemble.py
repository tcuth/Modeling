#!/home/llockhart/anaconda/bin/python

import numpy as np

from scipy.optimize import minimize
from scipy.stats import ks_2samp

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss

import pickle

import random

from optparse import OptionParser

usage = "%prog [options] TRAIN_LIBSVM MODEL_PKL [MODEL_PKL ...]"
parser = OptionParser(usage = usage)
parser.add_option("-s", "--scale", default = False, action = "store_true",\
		dest = "SCALE", help = "Min/Max scale data [default: %default]")
parser.add_option("-p","--proportion", default = 0.20, dest = "PROPORTION",\
		help = "Test split proportion [default: %default]")
options,args = parser.parse_args()

SCALE = bool(options.SCALE)
PROPORTION = float(options.PROPORTION)
DATA = args[0]
MODELS = args[1:]

X,y = load_svmlight_file(DATA)
X = X.toarray()

print "Creating train/test split."
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = PROPORTION)

if SCALE:
	print "Scaling data."
	scaler = MinMaxScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)


def ks(Y, preds):
	vals = np.unique(Y)
	pos_class = preds[:,1]
	return -1 * (ks_2samp(pos_class[Y == vals[0]], pos_class[Y == vals[1]])[0])

print "Importing models."
# put models in a list
clfs = []
for clf in MODELS:
	with open(clf,'r') as m:
		clfs.append(pickle.load(m))

print "Finding optimum weights."
### finding the optimum weights
predictions = []
for clf in clfs:
	predictions.append(clf.predict_proba(X_train))

def log_loss_func(weights):
	''' scipy minimize will pass the weights as a numpy array '''
	final_prediction = 0
	for weight, prediction in zip(weights, predictions):
		final_prediction += weight*prediction
		
	return log_loss(Y_train, final_prediction)

def ks_loss_func(weights):
	''' scipy minimize will pass the weights as a numpy array '''
	final_prediction = 0
	for weight, prediction in zip(weights, predictions):
		final_prediction += weight*prediction
		
	return ks(Y_train, final_prediction)
	

#the algorithms need a starting value, right not we chose 0.5 for all weights
#its better to choose many random starting points and run minimize a few times
#starting_values = [0.1]*len(predictions)
# random starting values
starting_values = [random.random() for i in predictions]

#adding constraints  and a different solver as suggested by user 16universe
#https://kaggle2.blob.core.windows.net/forum-message-attachments/75655/2393/otto%20model%20weights.pdf?sv=2012-02-12&se=2015-05-03T21%3A22%3A17Z&sr=b&sp=r&sig=rkeA7EJC%2BiQ%2FJ%2BcMpcA4lYQLFh6ubNqs2XAkGtFsAv0%3D
cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
#our weights are bound between 0 and 1
bounds = [(0,1)]*len(predictions)


best_score = float('Inf')
best_weights = []
for iter in range(1000):
	starting_values = [random.random() for i in predictions]
	res = minimize(ks_loss_func, starting_values, method='SLSQP', bounds=bounds, constraints=cons)
	if res['fun'] < best_score:
		best_score = res['fun']
		best_weights = res['x']
		print "Iteration: ", iter
		print "Best score so far: ", best_score
		print "Best weights so far: ", best_weights

print('Ensamble Score: {best_score}'.format(best_score=best_score))
print('Best Weights: {weights}'.format(weights=best_weights))

# evaluate weights on test set
predictions = []
for clf in clfs:
	predictions.append(clf.predict_proba(X_test))
final_prediction = 0
for weight, prediction in zip(best_weights, predictions):
	final_prediction += weight*prediction

print "Test Score: ", ks(Y_test, final_prediction)
