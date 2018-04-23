#!/usr/bin/env python

# Gradient Boosted Tree Classification

import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from hyperopt import hp, fmin, tpe, rand
from math import log

from argparse import ArgumentParser
from parseFields import parseFields
from time import time, strftime
import json
import pickle as pkl
from os.path import abspath

parser = ArgumentParser(description='A tool for running gradient boosted tree classification')

parser.add_argument('train')
parser.add_argument('test')
parser.add_argument('-x','--x_vars',default=None,type=str,help='Independent (X) variable columns[Linux-style parsing]')
parser.add_argument('-y','--y_var',default=None,type=int,help='Dependent (Y) variable column')
parser.add_argument('-w','--weights',default=None,type=int,help='Weights column (if present)')
parser.add_argument('-k','--k_folds',default=1,type=int,help='Number (k) of folds for cross_validation [default: 1]')
parser.add_argument('-e','--evals',default=None,type=int,help='Number of evaluations for hyperparameter search [default: None]')
parser.add_argument('-d','--delimiter',default='|',type=str,help='Input file delimiter [default: "|"]')
parser.add_argument('-s','--seed',default=2813308004,type=int,help='Random seed [default: 2813308004]')
parser.add_argument('-o','--output',default=None,type=str,help='Save final model to pkl [default: None]')
#parser.add_argument('-m','--metric',help='metric to minimize')
parser.add_argument('-c','--config',default=None,help='Configuration file (overrides command line options) [default: None]')
parser.add_argument('-z','--missing',default=-999999,type=float,help='Special values to treat as missing data [default: -999999]')
parser.add_argument('-a','--early',default=None,type=int,help='Early stopping iterations [default: None]')

args = parser.parse_args()

#METRIC = 'metric'
METRIC = 'error'
metric_func = None

def load_data(data_file, x_vars, y_var, weights):
	if x_vars is None or y_var is None:
		print "Must provide X and Y vars"
		sys.exit(1)
	else:
		x_vars = parseFields(x_vars)
		y_var = y_var - 1
		data = pd.read_csv(open(data_file), delimiter=args.delimiter)
		index = 0
		if weights is not None:
			index += 1
			weights = weights - 1
			data = data[[weights] + [y_var] + x_vars]
		else:
			data = data[[y_var] + x_vars]
		header = list(data.columns)[1+index:]
		target = list(data.columns)[index]
		XY = data.as_matrix()
		Y = XY[:,index]
		X = XY[:,index+1:]
		W = XY[:,0] if weights else np.ones(XY.shape[0])
		return X,Y,W,target,header

def train_fold(estimator, X, Y, W, train_index, test_index, metric):
	estimator.fit(X[train_index,:], Y[train_index])
	y_pred = estimator.predict_proba(X[test_index,:])
	return metric(Y[test_index], y_pred[:,1], W[test_index])

def cv_score(estimator, X, Y, W, k, metric):
	cv = StratifiedKFold(Y, n_folds = k, shuffle = True, random_state = args.seed)
	scores = [train_fold(estimator, X, Y, W, train_index, test_index, metric) for train_index, test_index in cv]
	return np.array(scores)

# Function to fit one iteration given a set of hyperparameters
# called in run_wrapper
def run_test(parameters):

	learning_rate, n_estimators, max_depth, subsample, min_samples_leaf = parameters

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
#   consider adding min_child_weight, gamma, and colsample_bytree

	parameters = {"eta": learning_rate, "n_estimators": n_estimators, "max_depth": max_depth,\
		"subsample": subsample, "min_child_weight": min_samples_leaf, \
		"colsample_by_tree": 1, "lambda": 0, "alpha": 0, "silent": 1, \
		"objective":"binary:logistic", 'base_score':BASE_SCORE, "missing":args.missing}
	scores = xgb.cv(parameters, dtrain, num_boost_round = n_estimators, nfold = args.k_folds, \
			seed = args.seed, early_stopping_rounds = args.early, metrics = {}, as_pandas = False, \
			verbose_eval = True, feval = metric_func)

	return scores

def run_wrapper(parameters):
	global RUN_COUNTER
	global BEST_SCORE

	RUN_COUNTER += 1
	print 'run', RUN_COUNTER

	start = time()
	scores = run_test(parameters)
	score = scores['train-error-mean'][-1]
	score_sd = scores['train-error-std'][-1]
	# update best score
	if score < BEST_SCORE:
		BEST_SCORE = score

	print
	print "%s: %.3f +/- %.3f" % (METRIC, score, score_sd)
	print "Elapsed: ", int(round(time()- start))
	print
	print "Best score: %.3f" % BEST_SCORE
	print "="*40

	return score

### BUILD MODEL ###

# Collect metadata if output is specified
if args.output:
	model_type = raw_input("Input model type.\n")
	model_type = model_type.strip()
	model_desc = raw_input("Provide breif model description.\n")
	model_desc = model_desc.strip()

print "Loading data."
X_train,Y_train,W_train,target_train,header_train = load_data(args.train,args.x_vars,args.y_var,args.weights)
dtrain = xgb.DMatrix(X_train,label=Y_train,missing=args.missing,weight=None,silent=False,feature_names = header_train)
X_test,Y_test,W_test,target_test,header_test = load_data(args.test,args.x_vars,args.y_var,args.weights)
dtest = xgb.DMatrix(X_test,label=Y_test,missing=args.missing,weight=None,silent=False,feature_names = header_test)

assert(target_train == target_test)
assert(header_train == header_test)
print "Training data contains %i rows and %i features." % X_train.shape
print "Test data contains %i rows and %i features." % X_test.shape
print "Metric to optimize: %s" % METRIC
print "Missing values: %s" % str(args.missing)

# use training set mean target rate as baseline score
# keeps prediction distribution in line with sklearn output
BASE_SCORE = np.mean(Y_train)

# run hyperopt if max_evals provided
# this saves parameters in dictionary called BEST
if args.evals:
	# hyperparameter search space
	space = (hp.loguniform('learning_rate', log(1e-3), log(0.2)),\
			hp.qlognormal('n_estimators', log(2000), 0.5, 100),\
			hp.quniform('max_depth', 3, 8, 1),\
			hp.uniform('subsample',0.5,1),\
			hp.qlognormal('min_samples_leaf', log(100), 0.5, 1)\
			)

	RUN_COUNTER = 0
	BEST_SCORE = np.inf
	start_time = time()
	BEST = fmin(run_wrapper, space, algo = rand.suggest, max_evals = args.evals, rstate = np.random.RandomState(args.seed))

	# End hyperopt
	print "Seconds passed:", int(round(time() - start_time))
	print BEST
	print "Score for best hyperparameters: %f.3" % BEST_SCORE

# BEST dictionary contains parameters from hyperopt or config, whichever is provided
# train final model if output provided
if args.output:
	print "Training model with best hyperparameters"
	learning_rate = float(BEST['learning_rate'])
	n_estimators = int(BEST['n_estimators'])
	max_depth = int(BEST['max_depth'])
	subsample = float(BEST['subsample'])
	min_samples_leaf = int(BEST['min_samples_leaf'])

	# train BEST hyperopt model
	print "Learning rate:", learning_rate
	print "N estimators:", n_estimators
	print "Max depth:", max_depth
	print "Subsample rate:", subsample
	print "Min samples per leaf:", min_samples_leaf

	parameters = {"eta": learning_rate, "n_estimators": n_estimators, "max_depth": max_depth,\
		"subsample": subsample, "min_child_weight": min_samples_leaf, \
		"colsample_by_tree": 1, "lambda": 0, "alpha": 0, "silent": 1, \
		"objective": "binary:logistic", 'base_score':BASE_SCORE,\
		"missing":args.missing}

	watchlist  = [(dtest,'eval'), (dtrain,'train')]
	evals = {}
	gbt = xgb.train(parameters, dtrain, n_estimators, watchlist, feval = metric_func, evals_result = evals)

	# calculate cross-validation metrics if config option provided
	if args.config:
		if cross_validate:
			print "Calculating cross-validated %s" % METRIC
			scores = xgb.cv(parameters, dtrain, num_boost_round = n_estimators, nfold = args.k_folds, seed = args.seed, early_stopping_rounds = args.early, \
						metrics = {}, as_pandas = False, verbose_eval = True, feval = metric_func)
			BEST_SCORE = scores['train-error-mean'][-1]
		else:
			BEST_SCORE = None

	metric_train = evals['train']['error'][gbt.best_iteration]
	metric_test = evals['eval']['error'][gbt.best_iteration]

	# save model to pkl object
	if args.output.split('.')[-1] != "pkl":
		args.output = args.output + ".pkl"
	print "Saving model to %s" %args.output
	output = {}
	output['model'] = gbt
	output['metric'] = METRIC
	output['cv_score'] = np.abs(BEST_SCORE) if BEST_SCORE else None
	output['train_score'] = metric_train
	output['test_score'] = metric_test
	output['best_iteration'] = gbt.best_iteration
	output['learning_curve'] = evals
	output['parameters'] = BEST
	output['creation_date'] = strftime("%c")
	output['training_data'] = abspath(args.train)
	output['test_data'] = abspath(args.test)
	output['variables'] = header_train
	output['model_type'] = model_type
	output['model_description'] = model_desc
	output['target'] = target_train
	output['configuration'] = {}
	if args.config:
		output['configuration']['run_type'] = 'config file'
		output['configuration']['options'] = config
	else:
		output['configuration']['run_type'] = 'hyperopt'
		output['configuration']['options'] = {}
		output['configuration']['options']['options'] = vars(args)
	with open(args.output,'w') as m:
		pkl.dump(output,m)
