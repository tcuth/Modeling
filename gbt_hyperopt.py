#!/home/llockhart/anaconda/bin/python

### SGD Classifier tuned with hyperopt

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import sys
import csv
import pickle
from time import time
from math import log

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import make_scorer, log_loss
from sklearn.metrics.scorer import check_scoring
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.datasets import load_svmlight_files, load_svmlight_file
from scipy.sparse import hstack
from sklearn.preprocessing import MinMaxScaler

from joblib import Parallel, delayed

from optparse import OptionParser

from hyperopt import hp, fmin, tpe

from weighted_ks import calcKS

usage = "%prog [options] INPUT_LIBSVM"
parser = OptionParser(usage = usage)
parser.add_option("-s", "--scale", default = False, action = "store_true",\
		dest = "SCALE", help = "Min/Max scale data [default: %default]")
parser.add_option("-k","--kfolds", default = 3, dest = "K",\
		help = "Number of CV folds [default: %default]")
parser.add_option("-o", "--output", default = "output.psv", dest = "OUTPUT",\
		help = "Save log to file [default: %default].")
parser.add_option("-e","--evals", default = 50, dest = "MAX_EVALS",\
		help = "Number of evaluations [default: %default]")
parser.add_option("-r","--random-seed", default = 8675309, action = "store",\
		dest = "SEED", help = "Random seed [default: %default]")
parser.add_option("-w","--weights", help = "Weights column (zero indexed, if present) [default: %default].",\
                dest = "WEIGHTS", default = None)
parser.add_option("-n","--njobs", default = 4, dest = "NJOBS",\
		help = "Number of parallel CV jobs.")
options,args = parser.parse_args()

# save command line options as variables
SCALE = options.SCALE
K = int(options.K)
OUTPUT = options.OUTPUT
MAX_EVALS = int(options.MAX_EVALS)
NJOBS = int(options.NJOBS)
RUN_COUNTER = 0
SEED = int(options.SEED)
if options.WEIGHTS is not None:
	WEIGHTS = int(options.WEIGHTS)
else:
	WEIGHTS = None

# check number of arguments
if len(args) != 1:
	print "Must contain exactly one arguments."
	sys.exit(1)

# load data
print "Loading data."
DATA = args[0]
X_train,Y_train = load_svmlight_file(DATA)
X_train = X_train.toarray()

# min-max scale data if requested
if SCALE:
	print "Scaling data."
	scaler = MinMaxScaler()
	X_train = scaler.fit_transform(X_train)

# set weights for KS calculation
if WEIGHTS is not None:
	weights = X_train[:,WEIGHTS]
	X_left = X_train[:, :WEIGHTS]
	X_right = X_train[:, WEIGHTS + 1:]
	X_train = np.concatenate((X_left,X_right),axis = 1)
else:
	weights = np.ones(X_train.shape[0])

# KS function
# Hyeropt requires smaller is better objective function
# Mutliply KS by -1 to acheive this
def absKS(tag, score, weights):
	return -1 * calcKS(tag, score, weights)

# Function to train one CV fold
# This allows for parallelization below
def train_one_fold(estimator, X, y, w, train_idx, test_idx):
	estimator.fit(X[train_idx,:],y[train_idx])
	y_pred = estimator.predict_proba(X[test_idx,:])
	return absKS(y[test_idx], y_pred[:,1], weights[test_idx])


# Parallel CV function
# Hacked from cross_val_score function in sklearn.cross_validation
# Trains all folds in parallel
def cv_score(estimator, X, y, weights,n_jobs,k):
	cv = StratifiedKFold(y, n_folds = k, shuffle = True)
	parallel = Parallel(n_jobs=n_jobs)
	scores = parallel(delayed(train_one_fold)(estimator, X, y, weights, train_idx, test_idx) \
			for train_idx, test_idx in cv)
	return np.array(scores)
# ------------------------------------------------------

# function to run one iteration of hyperopt and print stats to screen
def run_wrapper(params):
	global RUN_COUNTER
	global o_f

	RUN_COUNTER += 1
	print "run", RUN_COUNTER

	s = time()
	scores = run_test(params)
	score = np.mean(scores)

	print
	print "KS: ", np.mean(scores)
	print "Scores: ", scores
	print "Elapsed: {}s \n".format(int(round(time() - s)))

	writer.writerow([score] + list(params))
	o_f.flush()

	return np.mean(scores)

# Function to fit one CV iteration given a set of hyperparameters
# This is called in run_wrapper above
def run_test(params):

	learning_rate, n_estimators, max_depth, subsample, min_samples_leaf = params

	learning_rate = float(learning_rate)
	n_estimators = int(n_estimators)
	max_depth = int(max_depth)
	subsample = float(subsample)
	min_samples_leaf = int(min_samples_leaf)

	print "Learning rate:", learning_rate
	print "N estimators:", n_estimators
	print "Max depth:", max_depth
	print "Subsample rate:", subsample
	print "Min samples per leaf:", min_samples_leaf

	gbt = GradientBoostingClassifier(learning_rate = learning_rate, n_estimators = n_estimators, \
			max_depth = max_depth, subsample = subsample, min_samples_leaf = min_samples_leaf, \
			loss = 'deviance', max_features = 'auto', verbose = 0.0, random_state = SEED)
	scores = cv_score(gbt, X_train, Y_train, weights, n_jobs = NJOBS, k = K)

	return scores

# Hyperparameter space
space = (hp.loguniform('learning_rate', log(1e-3), log(10)),\
		hp.quniform('n_estimators', 1, 2000, 1),\
		hp.quniform('max_depth', 1, 10, 1),\
		hp.uniform('subsample',0,1),\
		hp.quniform('min_samples_leaf', 1, 100, 1)\
		)

# Print header row of output to file
headers = ['score','learning_rate','n_estimators','max_depth','subsample','min_samples_leaf']
o_f = open(OUTPUT,'wb')
writer = csv.writer(o_f, delimiter = "|")
writer.writerow(headers)

# Start hyperopt run
start_time = time()
best = fmin(run_wrapper, space, algo = tpe.suggest, max_evals = MAX_EVALS, rseed = SEED)
end_time = time()

# End hyperopt
print "Seconds passed:", int(round(end_time - start_time))
print best

