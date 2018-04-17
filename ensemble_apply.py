#!/home/llockhart/anaconda/bin/python

import numpy as np

import sys

import pandas as pd

from scipy.optimize import minimize
from scipy.stats import ks_2samp

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import log_loss

import pickle

import random

from optparse import OptionParser

usage = "%prog [options] DATA_LIBSVM MODEL_PKL [MODEL_PKL ...] > PREDICTIONS "
parser = OptionParser(usage = usage)
parser.add_option("-s", "--scale", default = False, action = "store_true",\
		dest = "SCALE", help = "Min/Max scale data [default: %default]")
parser.add_option("-w","--weights", default = None, dest = "WEIGHTS",\
		help = "List of weights [default: %default]")
options,args = parser.parse_args()

SCALE = bool(options.SCALE)
WEIGHTS = options.WEIGHTS.strip().split(",")
WEIGHTS = map(float, WEIGHTS)
DATA = args[0]
MODELS = args[1:]

assert len(WEIGHTS) == len(MODELS)

X,y = load_svmlight_file(DATA)
X = X.toarray()
X_train = X

if SCALE:
	scaler = MinMaxScaler()
	X_train = scaler.fit_transform(X_train)


def ks(Y, preds):
	vals = np.unique(Y)
	pos_class = preds[:,1]
	return -1 * (ks_2samp(pos_class[Y == vals[0]], pos_class[Y == vals[1]])[0])

# put models in a list
clfs = []
for clf in MODELS:
	with open(clf,'r') as m:
		clfs.append(pickle.load(m))


# calculate weighted predictions
predictions = []
for clf in clfs:
	predictions.append(clf.predict_proba(X_train))
final_prediction = 0
for weight, prediction in zip(WEIGHTS, predictions):
	final_prediction += weight*prediction

out = pd.DataFrame(final_prediction)
out.to_csv(sys.stdout, sep = "|", header = ['pred0','pred1'], index = False)
