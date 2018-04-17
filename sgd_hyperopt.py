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

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import make_scorer, log_loss
from sklearn.cross_validation import cross_val_score
from sklearn.datasets import load_svmlight_files, load_svmlight_file
from scipy.sparse import hstack
from sklearn.preprocessing import MinMaxScaler

from optparse import OptionParser

from hyperopt import hp, fmin, tpe


usage = "%prog [options] INPUT_LIBSVM"
parser = OptionParser(usage = usage)
parser.add_option("-s","--scale", default = False, action = "store_true",\
		dest = "SCALE", help = "Min/Max scale data [default: %default]")
parser.add_option("-k","--kfolds", default = 3, dest = "K",\
		help = "Number of CV folds [default: %default]")
parser.add_option("-o", "--output", default = "output.psv", dest = "OUTPUT",\
		help = "Save log to file [default: %default].")
parser.add_option("-e","--evals", default = 50, dest = "MAX_EVALS",\
		help = "Number of evaluations [default: %default]")
parser.add_option("-r","--random-seed", default = 8675309, dest = "SEED",\
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

	loss, penalty, alpha, l1_ratio = params

	loss = str(loss)
	penalty = str(penalty)
	alpha = float(alpha)
	l1_ratio = float(l1_ratio)

	print "Loss: ", loss
	print "Penalty: ", penalty
	print "Alpha: ", alpha
	print "L1 Ratio", l1_ratio

	sgd = SGDClassifier(loss = loss, penalty = penalty, alpha = alpha, l1_ratio = l1_ratio,\
		n_jobs = -1, n_iter=200, random_state = SEED)

	scores = cross_val_score(sgd, X_train, Y_train, cv = K, scoring = ks_scorer)

	return scores.mean()


space = (hp.choice('loss',['log','modified_huber']),\
		hp.choice('penalty',['l1','l2','elasticnet']),\
		hp.uniform('alpha', 0, 0.1),\
		hp.uniform('l1_ratio', 0, 1),\
		)


headers = ['score','loss','penalty','alpha','l1_ratio']
o_f = open(OUTPUT,'wb')
writer = csv.writer(o_f, delimiter = "|")
writer.writerow(headers)

start_time = time()
best = fmin(run_wrapper, space, algo = tpe.suggest, max_evals = MAX_EVALS)
end_time = time()

print "Seconds passed:", int(round(end_time - start_time))
print best

