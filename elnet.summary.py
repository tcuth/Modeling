#!/data/bin/anaconda/bin/python

import pickle
import csv
import pandas as pd
import numpy as np
from optparse import OptionParser
import sys


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
print "\t Alpha: %.5f" % modelDict['params']['alpha']
print "\t L1 ratio: %.5f" % modelDict['params']['l1_ratio']
print "\t Learning rate (eta): %.5f" % modelDict['params']['eta']
print ""
print "alpha = a + b,  L1 ratio = __a__,  penalty = a * L1 + b * L2"
print "\t\t\t   a + b"
print "a = %.9f" % (modelDict['params']['alpha'] * modelDict['params']['l1_ratio'])
print "b = %.9f" % (.5*modelDict['params']['alpha'] * (1-modelDict['params']['l1_ratio']))
