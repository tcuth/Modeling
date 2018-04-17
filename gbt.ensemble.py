#!/home/llockhart/anaconda/bin/python

import numpy as np

from scipy.optimize import minimize

from weighted_ks import calcKS
from load_data import load_data

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import log_loss

import pickle

import random

from optparse import OptionParser

usage = "%prog [options] TRAIN_PSV MODEL_PKL [MODEL_PKL ...]"
parser = OptionParser(usage = usage)
parser.add_option("-p","--proportion", default = 0.20, dest = "PROPORTION",\
		help = "Test split proportion [default: %default]")
options,args = parser.parse_args()

PROPORTION = float(options.PROPORTION)
DATA = args[0]
MODELS = args[1:]

# import models:
print "Importing models."
# put models in a list
clfs = []
for clf in MODELS:
	with open(clf,'r') as m:
		clfs.append(pickle.load(m))

# check that models were trained on same data
training_data = [clf['training_data'] for clf in clfs]
#assert(len(set(training_data)) == 1)

# load data
print "Loading data."
X,Y,weight,dv,header = load_data(clfs[0]['training_data'],\
		clfs[0]['configuration']['options']['options']['X_VARS'],\
		clfs[0]['configuration']['options']['options']['Y_VAR'],\
		clfs[0]['configuration']['options']['options']['WEIGHTS'])

print "Creating train/test split."
sss = StratifiedShuffleSplit(Y, 1, test_size = PROPORTION)
for i,j in sss:
	train_idx = i
	test_idx = j

X_train = X[train_idx,:]
Y_train = Y[train_idx]
if weight is not None:
	weights_train = weight[train_idx]
else:
	weights = np.ones(len(Y_train))

X_test = X[test_idx,:]
Y_test = Y[test_idx]
if weight is not None:
	weights_test = weight[test_idx]
else:
	weights = np.ones(len(Y_test))

print "Finding optimum weights."
### finding the optimum weights
predictions = []
for clf in clfs:
	predictions.append(clf['model'].predict_proba(X_train)[:,1])

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
		
	return -1 * calcKS(Y_train, final_prediction, weights_train)
	

#adding constraints  and a different solver as suggested by user 16universe
#https://kaggle2.blob.core.windows.net/forum-message-attachments/75655/2393/otto%20model%20weights.pdf?sv=2012-02-12&se=2015-05-03T21%3A22%3A17Z&sr=b&sp=r&sig=rkeA7EJC%2BiQ%2FJ%2BcMpcA4lYQLFh6ubNqs2XAkGtFsAv0%3D
cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
#our weights are bound between 0 and 1
bounds = [(0,1)]*len(predictions)

#import pdb
#pdb.set_trace()

best_score = float('Inf')
best_weights = []
for iter in range(100):
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
	predictions.append(clf['model'].predict_proba(X_test)[1])
final_prediction = 0
for weight, prediction in zip(best_weights, predictions):
	final_prediction += weight*prediction

print "Test Score: ", calcKS(Y_test, final_prediction, weights_test)
