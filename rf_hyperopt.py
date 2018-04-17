#!/home/llockhart/anaconda/bin/python

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
parser.add_option("-r", "--random-seed", default = 8675309, dest = "SEED",\
		help = "Random seed [default: %default]")
options,args = parser.parse_args()

SCALE = options.SCALE
K = int(options.K)
OUTPUT = options.OUTPUT
MAX_EVALS = int(options.MAX_EVALS)
SEED = int(options.SEED)
RUN_COUNTER = 0

if len(args) != 1:
	print "Must contain exactly one arguments."
	sys.exit(1)

print "Loading data."
DATA = args[0]
X_train,Y_train = load_svmlight_file(DATA)
X_train = X_train.toarray()


if SCALE:
	print "Scaling data."
	scaler = MinMaxScaler()
	X_train = scaler.fit_transform(X_train)

# KS function
def ks(Y, preds):
	vals = np.unique(Y)
	pos_class = preds[:,1]
	return -1 * (ks_2samp(pos_class[Y == vals[0]], pos_class[Y == vals[1]])[0])

ks_scorer = make_scorer(ks, greater_is_better = True, needs_proba = True)

# ------------------------------------------------------

def run_wrapper(params):
	global RUN_COUNTER
	global o_f

	RUN_COUNTER += 1
	print "run", RUN_COUNTER

	s = time()
	score = run_test(params)

	print
	print "KS: ", score
	print "Elapsed: {}s \n".format(int(round(time() - s)))

	writer.writerow([score] + list(params))
	o_f.flush()

	return score

def run_test(params):

	n_estimators, max_depth, min_samples_leaf = params

	n_estimators = int(n_estimators)
	max_depth = int(max_depth)
	min_samples_leaf = int(min_samples_leaf)

	print "N estimators:", n_estimators
	print "Max depth:", max_depth
	print "Min samples per leaf:", min_samples_leaf

	rf = RandomForestClassifier(n_estimators = n_estimators, criterion = 'gini',\
			max_depth = max_depth, min_samples_leaf = min_samples_leaf, \
			max_features = 'auto', verbose = 0.0, n_jobs = -1, random_state = SEED)

	scores = cross_val_score(rf, X_train, Y_train, cv = K, scoring = ks_scorer)

	return scores.mean()


space = (hp.quniform('n_estimators', 1, 2000, 1),\
		hp.quniform('max_depth', 1, 10, 1),\
		hp.qloguniform('min_samples_leaf', log(1), log(100), 1)\
		)


headers = ['score','n_estimators','max_depth','min_samples_leaf']
o_f = open(OUTPUT,'wb')
writer = csv.writer(o_f, delimiter = "|")
writer.writerow(headers)

start_time = time()
best = fmin(run_wrapper, space, algo = tpe.suggest, max_evals = MAX_EVALS)
end_time = time()

print "Seconds passed:", int(round(end_time - start_time))
print best

