#!/home/llockhart/anaconda/bin/python

### RandomForestClassifier tuned with hyperopt

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import sys
import csv
import pickle
from time import time
from math import log

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, log_loss
from sklearn.cross_validation import cross_val_score
from sklearn.datasets import load_svmlight_files, load_svmlight_file
from scipy.sparse import hstack
from sklearn.preprocessing import MinMaxScaler

from optparse import OptionParser

from hyperopt import hp, fmin, tpe


usage = "%prog [options] TRAIN_LIBSVM TEST_LIBSVM PARAMS_PSV"
parser = OptionParser(usage = usage)
parser.add_option("-s", "--scale", default = False, action = "store_true",\
		dest = "SCALE", help = "Min/Max scale data [default: %default]")
parser.add_option('-o', "--output", default = None, dest = "OUTPUT", \
		help = "File to save test predictions to (also saves model pickle). [default: %default]")
parser.add_option("-r","--random-state", default = 8675309, dest = "SEED",\
		help = "Random seed [default: %default]")
options,args = parser.parse_args()

SCALE = bool(options.SCALE)
OUTPUT = str(options.OUTPUT)
SEED = int(options.SEED)

if len(args) != 3:
	print "Must contain exactly three arguments."
	sys.exit(1)

print "Loading data."
TRAIN = args[0]
TEST = args[1]
X_train, Y_train, X_test, Y_test = \
	load_svmlight_files((TRAIN,TEST))
X_train = X_train.toarray()
X_test = X_test.toarray()
PARMS = pd.read_csv(args[2], sep = "|")

if SCALE:
	print "Scaling data."
	scaler = MinMaxScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)

# KS function
def ks(Y, preds):
	vals = np.unique(Y)
	pos_class = preds[:,1]
	return -1 * (ks_2samp(pos_class[Y == vals[0]], pos_class[Y == vals[1]])[0])

ks_scorer = make_scorer(ks, greater_is_better = True, needs_proba = True)

# ------------------------------------------------------
best = PARMS[PARMS.score == PARMS.score.min()]
best = best.to_dict('records')[0]

n_estimators = int(best['n_estimators'])
max_depth = int(best['max_depth'])
min_samples_leaf = int(best['min_samples_leaf'])

print "Training model with"
print "N estimators:", n_estimators
print "Max depth:", max_depth
print "Min samples per leaf:", min_samples_leaf

rf = RandomForestClassifier(n_estimators = n_estimators, criterion = 'gini',\
		max_depth = max_depth, min_samples_leaf = min_samples_leaf, \
		max_features = 'auto', verbose = 1.0, n_jobs = -1, random_state = SEED)
rf.fit(X_train, Y_train)

preds = rf.predict_proba(X_test)
preds2 = rf.predict_proba(X_train)

test_ks = ks_2samp(preds[Y_test == 0,1], preds[Y_test == 1, 1])[0]
train_ks = ks_2samp(preds2[Y_train == 0, 1], preds2[Y_train == 1, 1])[0]
print "Test KS: %s" % str(test_ks)
print "Train KS: %s" % str(train_ks)
print "Done training RandomForestClassifier."

if OUTPUT:
	MODEL_FILE = OUTPUT.split(".")[0]
	with open(MODEL_FILE + '.pkl','w') as m:
		pickle.dump(rf,m)
	pd.DataFrame(np.hstack((preds,Y_test.reshape(Y_test.shape[0],1)))).to_csv(OUTPUT,sep = '|', header = ['pred0','pred1','tag'], index = False)

