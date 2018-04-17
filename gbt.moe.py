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
import requests

from sklearn.ensemble import GradientBoostingClassifier
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

from weighted_ks import calcKS

from fdr import fraud_detection_rate

# import helpers directory
sys.path.append('/data/bin/')
from parseFields import parseFields

sys.path.append('/data/bin/modeling/')

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
	elif EXT == ".psv":
		if X_VARS is None or Y_VAR is None:
			print "Must provide X and Y vars with .psv file."
			sys.exit(1)
		else:
			X_VARS = parseFields(X_VARS)
			Y_VAR = int(Y_VAR) - 1
			XY = pd.read_csv(data, delimiter = "|")
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
		print "Invalid file extension: %s (must be .libsvm or .psv)" % EXT
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
			loss = 'deviance', max_features = 1.0, verbose = 0.0, random_state = SEED)
	scores = cv_score(gbt, X_train, Y_train, weights, n_jobs = NJOBS, k = K, metric = METRIC)

	return scores
# ------------------------------------------------------------

# save command line options/config options as variables
CONFIG = options.CONFIG

if CONFIG:
	with open(CONFIG,'r') as f:
		config = json.load(f)
	MAX_EVALS = None
	# manually set gbt hyperparameteres
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
print "\n"

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
	best_score = np.inf
	best = {}
	url = 'http://0.0.0.0:6543/gp/next_points/epi'
	headers = {'Content-Type': 'application/json'}
	with open('/data/bin/modeling/moe.config.json','r') as f:
		moe_params = json.load(f)
	
	for i in range(MAX_EVALS):
		r = requests.post(url, data = json.dumps(moe_params), headers = headers)
		learning_rate, n_estimators, max_depth, subsample, min_samples_leaf = json.loads(r.text)['points_to_sample'][0]
		learning_rate = float(learning_rate)
		n_estimators = int(float(n_estimators))
		max_depth = int(float(max_depth))
		subsample = float(subsample)
		min_samples_leaf = int(float(min_samples_leaf))

		hyperparameters = (learning_rate, n_estimators, max_depth, subsample, min_samples_leaf)
		scores = run_test(hyperparameters)
		update_point = {"point": list(hyperparameters),
						"value": np.mean(scores),\
						"value_var": np.var(scores)}
		moe_params["gp_historical_info"]["points_sampled"].append(update_point)
		print scores
		print "\n"
		if np.mean(scores) < best_score:
			best['learning_rate'], best['n_estimators'], best['max_depth'], best['subsample'], best['min_samples_leaf'] = hyperparameters
			best_score = np.mean(scores)


# best dict contains params from hyperopt or config, whichever is provided
# train final model if option provided
if OUTPUT:
# train with best hyperparamters
	print "Training model with best hyperparameters"
	learning_rate = float(best['learning_rate'])
	n_estimators = int(best['n_estimators'])
	max_depth = int(best['max_depth'])
	subsample = float(best['subsample'])
	min_samples_leaf = int(best['min_samples_leaf'])

	# train best hyperopt model
	print "Learning rate:", learning_rate
	print "N estimators:", n_estimators
	print "Max depth:", max_depth
	print "Subsample rate:", subsample
	print "Min samples per leaf:", min_samples_leaf

	gbt = GradientBoostingClassifier(learning_rate = learning_rate, n_estimators = n_estimators, \
			max_depth = max_depth, subsample = subsample, min_samples_leaf = min_samples_leaf, \
			loss = 'deviance', max_features = 1.0, verbose = 1.0, random_state = SEED)
	gbt.fit(X_train, Y_train)

	# calculate cv ks if config option provided
	if CONFIG:
		if CROSS_VALIDATE:
			print "Calculating cross-validated ks."
			n_jobs = 3
			k = 3
			scores =  cv_score(gbt, X_train, Y_train, weights, n_jobs, k, METRIC)
			best_score = np.mean(scores)
		else:
			best_score = None

	# KS information
	preds_train = gbt.predict_proba(X_train)[:,1]
	if METRIC == "KS":
		train_ks = calcKS(Y_train, preds_train, weights)
	elif METRIC == "FDR":
		train_ks = -1 * negFDR(Y_train, preds_train, weights, 95)
	else:
		print "Metric must be FDR or KS."
		sys.exit(1)
	if TEST:
		preds_test = gbt.predict_proba(X_test)[:,1]
		if METRIC == "KS":
			test_ks = calcKS(Y_test, preds_test, weights_test)
		elif METRIC == "FDR":
			test_ks = -1 * negFDR(Y_test, preds_test, weights_test, 95)


	if splitext(OUTPUT)[1] != ".pkl":
		OUTPUT = OUTPUT + ".pkl"
	print "Saving model to %s" % OUTPUT
	output = {}
	output['moe_history'] = moe_params
	output['model'] = gbt
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
