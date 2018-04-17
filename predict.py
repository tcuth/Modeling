#!/home/llockhart/anaconda/bin/python

import sys
from sklearn.datasets import load_svmlight_file
import pickle
from optparse import OptionParser
import pandas as pd
import numpy as np

usage = "Usage: %prog INPUT_LIBSVM INPUT_MODEL_PKL OUTPUT_PREDS"
parser = OptionParser(usage = usage)
parser.add_option("-w","--weights",default = None, dest = "WEIGHTS",\
		help = "Weights column (zero indexed, if present) [default: %default]")
options,args = parser.parse_args()

if options.WEIGHTS is not None:
	WEIGHTS = int(options.WEIGHTS)
else:
	WEIGHTS = None

INPUT_FILE = args[0]
MODEL_FILE = args[1]
OUTPUT_FILE = args[2]

# for testing purposes
#INPUT_FILE = 'xorvars_credit_woe.libsvm'
#MODEL_FILE = 'sgd_woe_credit.pkl'
#OUTPUT_FILE = 'tmppreds.psv'

print "Loading data."
X,Y = load_svmlight_file(INPUT_FILE)
X = X.toarray()

if WEIGHTS is not None:
	weights = X[:,WEIGHTS]
	X_left = X[:, :WEIGHTS]
	X_right = X[:, WEIGHTS + 1:]
	X = np.concatenate((X_left,X_right),axis = 1)

print "Loading model."
with open(MODEL_FILE,'r') as m:
	model = pickle.load(m)

print "Creating predictions."
preds = model.predict_proba(X)

print "Writing predictions to %s." % OUTPUT_FILE
out = pd.DataFrame(preds)
out.to_csv(OUTPUT_FILE, sep = '|', header = ['pred0','pred1'], index = False)
