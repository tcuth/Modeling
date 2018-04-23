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
print "Columns in file used in modeling: %s" % modelDict['configuration']['options']['options']['x_vars']
print "Training Data: %s" % modelDict['training_data']
print "Test Data: %s" % modelDict.get('test_data',None)
print "Target: %s" % modelDict['target']
print "Evaluation metric: %s" % modelDict.get('metric','error')
print "Best iteration %i" % modelDict['best_iteration']
print "CV %s: %.3f" % (modelDict.get('metric','error'), modelDict['cv_score'])
print "Training %s: %.3f" % (modelDict.get('metric','error'), modelDict['train_score'])
print "Test %s: %.3f" % (modelDict.get('metric','error'), modelDict['test_score'])
print "Hyperparameters"
print "\t Learning Rate: %.3f" % modelDict['parameters']['learning_rate']
print "\t Max Depth: %i" % modelDict['parameters']['max_depth']
print "\t Min Samples Leaf: %i" % modelDict['parameters']['min_samples_leaf']
print "\t Number of Trees: %i" % modelDict['parameters']['n_estimators']
print "\t Subsample Rate: %.3f" % modelDict['parameters']['subsample']

