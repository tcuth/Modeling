#!/usr/bin/env python

### XGB Classifier tuned with hyperopt

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import sys
import csv
import pickle
from time import time, strftime
from math import log
import json

import readline
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr

import xgboost as xgb
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

from hyperopt import hp, fmin, tpe, rand as random

from weighted_ks import calcKS

from fdr import fraud_detection_rate

sys.path.append('/data/bin')
from parseFields import parseFields

from collections import OrderedDict

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
parser.add_option("-z","--missing", default = -999, dest = "MISSING",\
		help = "Missing values [default: %default]")
parser.add_option("-a","--early-stopping-iterations", default = None, dest = "STOPPING",\
		help = "Early stopping iterations [default: %default]")
choices_t = OrderedDict([("random", random.suggest), ("tpe", tpe.suggest)])
parser.add_option("-t","--tuning", choices = choices_t.keys(), default = choices_t.keys()[0], dest = "TUNING",\
		help = "Search method for hyperparameter tuning [choices: %s, default: %s]" % (choices_t.keys(), choices_t.keys()[0]))
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
#def absKS(preds, dtrain):
#	y = dtrain.get_label()
#	ks = calcKS(y, preds)
#	return 'ks', -1 * ks
creditR = importr("creditR")
numpy2ri.activate()
def absKS(preds, dtrain):
	y = dtrain.get_label()
	ks = creditR.ksC(preds,y)
	return 'ks', -1.0 * ks[0]

# FDR function
def negFDR(preds, dtrain):
	y = dtrain.get_label()
	fdr = fraud_detection_rate(y, preds, None, 95)[0]
	return 'fdr', -1 * fdr

# put eval functions in dictionary for calling in xgb.train
eval_func = {'KS':absKS, 'FDR': negFDR}

def run_wrapper(params):
	global RUN_COUNTER
	global best_score

	RUN_COUNTER += 1
	print "run", RUN_COUNTER

	s = time()
	scores = run_test(params)
	score = scores[-1,0]
	score_sd = scores[-1,1]

	print
	print "%s: %.3f +/- %.3f" % (METRIC, score, score_sd)
	print "Elapsed: {}s \n".format(int(round(time() - s)))

	# update best score
	if score < best_score:
		best_score = score
	
	print "Best score: %.3f" % best_score
	print "=" * 40

	return score

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
#	consider adding min_child_weight, gamma, and colsample_bytree

	params = {"eta": learning_rate, "n_estimators": n_estimators, "max_depth": max_depth,\
		"subsample": subsample, "min_child_weight": min_samples_leaf, \
		"colsample_by_tree": 1, "lambda": 0, "alpha": 0, "silent": 1, "nthread": NJOBS,\
		"objective":"binary:logistic", 'base_score':BASE_SCORE, "missing":MISSING}
	scores = xgb.cv(params, dtrain, num_boost_round = n_estimators, nfold = K, seed = SEED, early_stopping_rounds = STOPPING, \
			metrics = {}, as_pandas = False, show_progress = True, feval = eval_func[METRIC])

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
	NJOBS = config['options'].get('NJOBS',1)
	SEED = config['options']['SEED']
	Y_VAR = config['options']['Y_VAR']
	X_VARS = config['options']['X_VARS']
	WEIGHTS = config['options']['WEIGHTS']
	METRIC = config['options'].get('METRIC','KS')
	if WEIGHTS:
		WEIGHTS = int(WEIGHTS) - 1
	CROSS_VALIDATE = config['options']['CROSS_VALIDATE']
	MISSING = config['options'].get('MISSING',None)
	DATA = config['data']['training']
	TEST = config['data']['test']
	STOPPING = config['options']['STOPPING']
	TUNING = config['options']['TUNING']
	if TUNING:
		TUNING = choices_t[TUNING]
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
	MISSING = float(options.MISSING) if options.MISSING is not None else None
	if options.STOPPING:
		STOPPING = int(options.STOPPING)
	else:
		STOPPING = None
	if len(args) not in [1,2]:
		print "Must contain exactly one or two arguments."
		sys.exit(1)
	DATA = args[0]
	if len(args) == 2:
		TEST = args[1]
	else:
		TEST = None
	TUNING = choices_t[options.TUNING]

# collect metadata if output requests
if OUTPUT:
	model_type = raw_input("Input model type (i.e., credit, fraud, etc.)\n" )
	model_type = model_type.strip()
	model_description = raw_input("Provide brief model description.\n")
	model_description = model_description.strip()

# load data
print "Loading data."
X_train,Y_train,weights,dv,header = load_data(DATA,X_VARS,Y_VAR,WEIGHTS)
dtrain = xgb.DMatrix(X_train, label = Y_train, missing = MISSING, weight = None, \
	silent = False, feature_names = header)
if TEST:
	X_test,Y_test,weights_test,dv_test,header_test = load_data(TEST,X_VARS,Y_VAR,WEIGHTS)
	dtest = xgb.DMatrix(X_test, label = Y_test, missing = MISSING, weight = None, \
		silent = False, feature_names = header_test)

	assert(dv == dv_test)
	assert(header == header_test)
print "Training data contains %i rows and %i features." % X_train.shape
if TEST:
	print "Test data contains %i rows and %i features." % X_test.shape
print "Metric to optimize: %s" % METRIC
print "Missing values: %s" % str(MISSING)
print "=" * 40

# use training set mean target rate as baseline score
# keeps prediction distribution in line with sklearn output
BASE_SCORE = np.mean(Y_train)

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
	space = (hp.loguniform('learning_rate', log(1e-3), log(0.2)),\
			hp.qlognormal('n_estimators', log(2000), 0.5, 100),\
			hp.quniform('max_depth', 3, 8, 1),\
			hp.uniform('subsample',0.5,1),\
			hp.qlognormal('min_samples_leaf', log(100), 0.5, 1)\
			)

	# Start hyperopt run
	RUN_COUNTER = 0
	best_score = np.inf
	start_time = time()
	best = fmin(run_wrapper, space, algo = TUNING, max_evals = MAX_EVALS, rseed = SEED)
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

	params = {"eta": learning_rate, "n_estimators": n_estimators, "max_depth": max_depth,\
		"subsample": subsample, "min_child_weight": min_samples_leaf, \
		"colsample_by_tree": 1, "lambda": 0, "alpha": 0, "silent": 1, \
		"objective": "binary:logistic", "nthread": NJOBS, 'base_score':BASE_SCORE,\
		"missing":MISSING}

	watchlist  = [(dtest,'eval'), (dtrain,'train')]
	gbt = xgb.train(params, dtrain, n_estimators, watchlist, feval = eval_func[METRIC])

	# calculate cv ks if config option provided
	if CONFIG:
		if CROSS_VALIDATE:
			print "Calculating cross-validated ks."
			params['nthread'] = NJOBS
			K = 3
			scores = xgb.cv(params, dtrain, num_boost_round = n_estimators, nfold = K, seed = SEED, early_stopping_rounds = STOPPING, \
				metrics = {}, as_pandas = False, show_progress = True, feval = eval_func[METRIC])
			
			best_score = scores[-1,0]
		else:
			best_score = None

	# KS information
	preds_train = gbt.predict(dtrain)
	if METRIC == "KS":
		train_ks = calcKS(Y_train, preds_train, weights)
	elif METRIC == "FDR":
		train_ks = fraud_detection_rate(Y_train, preds_train, weights, 95)[0]
	else:
		print "Metric must be FDR or KS."
		sys.exit(1)
	if TEST:
		preds_test = gbt.predict(dtest)
		if METRIC == "KS":
			test_ks = calcKS(Y_test, preds_test, weights_test)
		elif METRIC == "FDR":
			test_ks = fraud_detection_rate(Y_test, preds_test, weights_test, 95)[0]


	if splitext(OUTPUT)[1] != ".pkl":
		OUTPUT = OUTPUT + ".pkl"
	print "Saving model to %s" % OUTPUT
	output = {}
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

numpy2ri.deactivate()
