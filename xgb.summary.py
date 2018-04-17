#!/usr/bin/env python

import pickle
import csv
import pandas as pd
import numpy as np
from optparse import OptionParser
import sys
import xgboost as xgb

# options parser
## use new pkl object from gbt.py that has all necessary information in a dictionary
usage = "Usage: %prog [options] MODEL_PKL"
parser = OptionParser(usage = usage)
options,args = parser.parse_args()

MODEL_FILE = args[0]

with(open(MODEL_FILE,'r')) as f:
	modelDict = pickle.load(f)

model = modelDict['model']
features = modelDict['variables']

print
print
print "Created at: %s" % modelDict['creation_date']
print "Model Type: %s" % modelDict['model_type']
print "Model Description: %s" % modelDict['model_description']
print "Number of features: %i" % len(modelDict['variables'])
print "Columns in file used in modeling: %s" % modelDict['configuration']['options']['options']['X_VARS']
print "Training Data: %s" % modelDict['training_data']
if modelDict['testing_data']:
	print "Test Data: %s" % modelDict['testing_data']
else:
	print "Test Data: None"
print "Target: %s" % modelDict['target']
print "Evaluation metric: %s" % modelDict.get('metric','KS')
if modelDict['cv_ks']:
	print "CV %s: %.3f" % (modelDict.get('metric','KS'), modelDict['cv_ks'])
else:
	print "CV %s: None" % modelDict.get('metric','KS')
print "Training %s: %.3f" % (modelDict.get('metric','KS'), modelDict['train_ks'])
if modelDict['test_ks']:
	print "Test %s: %.3f" % (modelDict.get('metric','KS'), modelDict['test_ks'])
else:
	print "Test %s: None" % modelDict.get('metric','KS')
print "Hyperparameters"
print "\t Learning Rate: %.3f" % modelDict['params']['learning_rate']
print "\t Max Depth: %i" % modelDict['params']['max_depth']
print "\t Min Samples Leaf: %i" % modelDict['params']['min_samples_leaf']
print "\t Number of Trees: %i" % modelDict['params']['n_estimators']
print "\t Subsample Rate: %.3f" % modelDict['params']['subsample']

