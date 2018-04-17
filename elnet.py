#!/data/bin/anaconda/bin/python

### SGD Classifier tuned with hyperopt

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import sys
import csv
import pickle
from time import time, strftime
from math import log
import json

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import make_scorer, log_loss
from sklearn.metrics.scorer import check_scoring
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.datasets import load_svmlight_files, load_svmlight_file
from scipy.sparse import hstack
from sklearn.preprocessing import MinMaxScaler

import pickle as pkl

from os.path import splitext, abspath

from joblib import Parallel, delayed

from optparse import OptionParser

from hyperopt import hp, fmin, tpe

sys.path.append('/data/bin/modeling/')
from weighted_ks import calcKS
from fdr import fraud_detection_rate

# import helpers directory
sys.path.append('/data/bin/')
from parseFields import parseFields

usage = "%prog [options] TRAINING_DATA (.libsvm or .psv) [TESTING_DATA (.libsvm or .psv)]"
parser = OptionParser(usage = usage)
parser.add_option("-s", "--scale", default = False, action = "store_true",\
		dest = "SCALE", help = "Min/Max scale data [default: %default]")
parser.add_option("-k","--kfolds", default = 3, dest = "K",\
		help = "Number of CV folds [default: %default]")
parser.add_option("-o", "--output", default = None, dest = "OUTPUT",\
		help = "Save final model as pkl [default: %default].")
parser.add_option("-e","--evals", default = 50, dest = "MAX_EVALS",\
		help = "Number of evaluations [default: %default]")
parser.add_option("-r","--random-seed", default = 8675309, action = "store",\
		dest = "SEED", help = "Random seed [default: %default]")
parser.add_option("-x", "--x-variables", default = None, action = "store", dest = "X_VARS",\
		help = "Independent (X) variables if .psv file provided [default: %default]")
parser.add_option("-y", "--y-variable", default = None, action = "store", dest = "Y_VAR",\
		help = "Dependent (Y) variable if .psv file provided [default: %default]")
parser.add_option("-w","--weights", help = "Weights column (if present) [default: %default].",\
                dest = "WEIGHTS", default = None)
parser.add_option("-n","--njobs", default = 4, dest = "NJOBS",\
		help = "Number of parallel CV jobs.")
parser.add_option("-m","--metric", default = "KS", dest = "METRIC",\
		help = "Metric to optimize (KS or FDR) [default: %default]")
parser.add_option("-c","--configuration", default = None, dest = "CONFIG",\
		help = "Configuration file (overides command line defaults [default: %default]")
options,args = parser.parse_args()

# ------------------------------------------------------------
# helper functions

def load_data(data,X_VARS,Y_VAR,WEIGHTS):
	from os.path import splitext
	EXT = splitext(data)[1]
	if EXT == ".libsvm":
		X_train,Y_train = load_svmlight_file(data)
		X_train = X_train.toarray()
		header = None
		dv = None
		# set weights for KS calculation
		if WEIGHTS is not None:
			weights = X_train[:,WEIGHTS]
			X_left = X_train[:, :WEIGHTS]
			X_right = X_train[:, WEIGHTS + 1:]
			X_train = np.concatenate((X_left,X_right),axis = 1)
		else:
			weights = np.ones(X_train.shape[0])
		return X_train,Y_train,weights,dv,header
	elif EXT in [".psv",".csv"]:
		if X_VARS is None or Y_VAR is None:
			print "Must provide X and Y vars with .psv file."
			sys.exit(1)
		else:
			X_VARS = parseFields(X_VARS)
			Y_VAR = int(Y_VAR) - 1
			if EXT == ".psv":
				XY = pd.read_csv(data, delimiter = "|")
			else:
				XY = pd.read_csv(data, delimiter = ",")
			if WEIGHTS:
				# reorder columns
				XY = XY[[Y_VAR] + [WEIGHTS] + X_VARS]
				header = list(XY.columns)[2:]
				dv = list(XY.columns)[0]
				XY = XY.as_matrix()
				Y_train = XY[:,0]
				weights = XY[:,1]
				X_train = XY[:,2:]
			else:
				XY = XY[[Y_VAR] + X_VARS]
				header = list(XY.columns)[1:]
				dv = list(XY.columns)[0]
				XY = XY.as_matrix()
				Y_train = XY[:,0]
				X_train = XY[:,1:]
				weights = np.ones(X_train.shape[0])
		return X_train,Y_train,weights,dv,header
	else:
		print "Invalid file extension: %s (must be .libsvm, .psv, or .csv)" % EXT
		sys.exit(1)

# KS function
# Hyeropt requires smaller is better objective function
# Mutliply KS by -1 to acheive this
def absKS(tag, score, weights):
	return -1 * calcKS(tag, score, weights)

# FDR function
def negFDR(tag, score, weights, p):
	return -1 * fraud_detection_rate(tag, score, weights, p)[0]

# Function to train one CV fold
# This allows for parallelization below
def train_one_fold(estimator, X, y, w, train_idx, test_idx, metric):
	estimator.fit(X[train_idx,:],y[train_idx])
	y_pred = estimator.predict_proba(X[test_idx,:])
	if metric == "KS":
		return absKS(y[test_idx], y_pred[:,1], weights[test_idx])
	elif metric == "FDR":
		return negFDR(y[test_idx], y_pred[:,1], weights[test_idx], 95)
	else:
		print "Metric must be either KS or FDR."
		sys.exit(1)

# Parallel CV function
# Hacked from cross_val_score function in sklearn.cross_validation
# Trains all folds in parallel
def cv_score(estimator, X, y, weights,n_jobs,k,metric):
	cv = StratifiedKFold(y, n_folds = k, shuffle = True, random_state = SEED)
	parallel = Parallel(n_jobs=n_jobs)
	scores = parallel(delayed(train_one_fold)(estimator, X, y, weights, train_idx, test_idx,metric) \
			for train_idx, test_idx in cv)
	return np.array(scores)

def run_wrapper(params):
	global RUN_COUNTER
	global best_score

	RUN_COUNTER += 1
	print "run", RUN_COUNTER

	s = time()
	scores = run_test(params)
	score = np.mean(scores)

	print
	print "%s: %.3f" % (METRIC,  np.mean(scores))
	print "Scores: ", scores
	print "Elapsed: {}s \n".format(int(round(time() - s)))

	# update best score
	if score < best_score:
		best_score = score
	
	print "Best score: %.3f" % best_score

	return np.mean(scores)

# Function to fit one CV iteration given a set of hyperparameters
# This is called in run_wrapper above
def run_test(params):

	alpha, l1_ratio, eta = params

        alpha = float(alpha)
        l1_ratio = float(l1_ratio)
        eta = float(eta)

        print "Alpha:", alpha
        print "L1 ratio:", l1_ratio
        print "Learning rate (eta):", eta

	el_net = SGDClassifier(loss = 'log', penalty = 'elasticnet', alpha = alpha,
                l1_ratio = l1_ratio, fit_intercept = True, n_iter = 250,
                shuffle = True, random_state = SEED, verbose = 0, n_jobs = 1,
                learning_rate = 'constant', eta0 = eta, warm_start = True)
        scores = cv_score(el_net, X_train, Y_train, weights, n_jobs = NJOBS, k = K, metric = METRIC)

	return scores
# ------------------------------------------------------------

# save command line options/config options as variables
CONFIG = options.CONFIG

if CONFIG:
	with open(CONFIG,'r') as f:
		config = json.load(f)
	MAX_EVALS = None
	# manually set elastic net hyperparameteres
	best = config['hyperparameters']
	SCALE = config['options']['SCALE']
	OUTPUT = config['options']['OUTPUT']
	SEED = config['options']['SEED']
	Y_VAR = config['options']['Y_VAR']
	X_VARS = config['options']['X_VARS']
	WEIGHTS = config['options']['WEIGHTS']
	METRIC = config['options'].get('METRIC','KS')
	if WEIGHTS:
		WEIGHTS = int(WEIGHTS) - 1
	CROSS_VALIDATE = config['options']['CROSS_VALIDATE']
	DATA = config['data']['training']
	TEST = config['data']['test']
else:
	SCALE = options.SCALE
	K = int(options.K)
	if options.OUTPUT:
		OUTPUT = options.OUTPUT
	MAX_EVALS = int(options.MAX_EVALS)
	NJOBS = int(options.NJOBS)
	SEED = int(options.SEED)
	X_VARS = options.X_VARS
	Y_VAR = options.Y_VAR
	if options.WEIGHTS is not None:
		WEIGHTS = int(options.WEIGHTS) - 1
	else:
		WEIGHTS = None
	METRIC = str(options.METRIC)
	if len(args) not in [1,2]:
		print "Must contain exactly one or two arguments."
		sys.exit(1)
	DATA = args[0]
	if len(args) == 2:
		TEST = args[1]
	else:
		TEST = None

# collect metadata if output requests
if OUTPUT:
	model_type = raw_input("Input model type (i.e., credit, fraud, etc.)\n" )
	model_type = model_type.strip()
	model_description = raw_input("Provide brief model description.\n")
	model_description = model_description.strip()

# load data
print "Loading data."
X_train,Y_train,weights,dv,header = load_data(DATA,X_VARS,Y_VAR,WEIGHTS)
if TEST:
	X_test,Y_test,weights_test,dv_test,header_test = load_data(TEST,X_VARS,Y_VAR,WEIGHTS)
	assert(dv == dv_test)
	assert(header == header_test)
print "Training data contains %i rows and %i features." % X_train.shape
if TEST:
	print "Test data contains %i rows and %i features." % X_test.shape
print "Metric to optimize: %s" % METRIC


# min-max scale data if requested
if SCALE:
	print "Scaling data."
	scaler = MinMaxScaler()
	X_train = scaler.fit_transform(X_train)
	if TEST:
		X_test = scaler.transform(X_test)

# run hyperopt if MAX_EVALS provided on command line
# this saves params in best dict
if MAX_EVALS:
	# Hyperparameter space
	space = (hp.loguniform('alpha',log(0.00001),log(1)),\
                        hp.uniform('l1_ratio',0.01,1),\
                        hp.loguniform('eta', log(1e-3), log(1)),\
			)


	# Start hyperopt run
	RUN_COUNTER = 0
	best_score = np.inf
	start_time = time()
	best = fmin(run_wrapper, space, algo = tpe.suggest, max_evals = MAX_EVALS, rseed = SEED)
	end_time = time()

	# End hyperopt
	print "Seconds passed:", int(round(end_time - start_time))
	print best
	print "Score for best hyperparameters: %f.3" % best_score

# best dict contains params from hyperopt or config, whichever is provided
# train final model if option provided
if OUTPUT:
# train with best hyperparamters
	print "Training model with best hyperparameters"
        alpha = float(best['alpha'])
        l1_ratio = float(best['l1_ratio'])
        eta = float(best['eta'])

	# train best hyperopt model
        print "Alpha:", alpha
        print "L1 ratio:", l1_ratio
	print "Learning rate (eta):", eta

	el_net = SGDClassifier(loss = 'log', penalty = 'elasticnet', alpha = alpha,
                l1_ratio = l1_ratio, fit_intercept = True, n_iter = 250,
                shuffle = True, random_state = SEED, verbose = 0, n_jobs = 1,
                learning_rate = 'constant', eta0 = eta, warm_start = True)
	el_net.fit(X_train, Y_train)

	# calculate cv ks if config option provided
	if CONFIG:
		if CROSS_VALIDATE:
			print "Calculating cross-validated ks."
			n_jobs = 3
			k = 3
			scores =  cv_score(el_net, X_train, Y_train, weights, n_jobs, k, METRIC)
			best_score = np.mean(scores)
		else:
			best_score = None

	# KS information
	preds_train = el_net.predict_proba(X_train)[:,1]
	if METRIC == "KS":
		train_ks = calcKS(Y_train, preds_train, weights)
	elif METRIC == "FDR":
		train_ks = -1 * negFDR(Y_train, preds_train, weights, 95)
	else:
		print "Metric must be FDR or KS."
		sys.exit(1)
	if TEST:
		preds_test = el_net.predict_proba(X_test)[:,1]
		if METRIC == "KS":
			test_ks = calcKS(Y_test, preds_test, weights_test)
		elif METRIC == "FDR":
			test_ks = -1 * negFDR(Y_test, preds_test, weights_test, 95)


	if splitext(OUTPUT)[1] != ".pkl":
		OUTPUT = OUTPUT + ".pkl"
	print "Saving model to %s" % OUTPUT
	output = {}
	output['model'] = el_net
	output['metric'] = METRIC
	if best_score:
		output['cv_ks'] = np.abs(best_score)
	else:
		output['cv_ks'] = None
	output['train_ks'] = train_ks
	if TEST:
		output['test_ks'] = test_ks
	else:
		output['test_ks'] = None
	output['params'] = best
	output['creation_date'] = strftime("%c")
	output['training_data'] = abspath(DATA)
	if TEST:
		output['testing_data'] = abspath(TEST)
	else:
		output['testing_data'] = None
	output['variables'] = header
	output['model_type'] = model_type
	output['model_description'] = model_description
	output['target'] = dv
	if SCALE:
		output['scaler'] = scaler
	# vars function touns options to dictionary
	output['configuration'] = {}
	if CONFIG:
		output['configuration']['run_type'] = 'config file'
		output['configuration']['options'] = config
	else:
		output['configuration']['run_type'] = 'hyperopt'
		output['configuration']['options'] = {}
		output['configuration']['options']['options'] = vars(options)
	with open(OUTPUT,'w') as m:
		pkl.dump(output,m)
