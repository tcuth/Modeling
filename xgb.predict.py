#!/usr/bin/env python

import pickle
import csv
import pandas as pd
import numpy as np
from optparse import OptionParser
import sys
import xgboost as xgb
from load_data import load_data
sys.path.append('/data/bin/')
from parseFields import parseFields
# options parser
## use new pkl object from xgb.py that has all necessary information in a dictionary
usage = "Usage: cat INPUT_FILE | gbt.predict.py [options] MODEL_PKL > OUTPUT_FILE"

parser = OptionParser(usage = usage)
parser.add_option("-k","--keep", dest = "KEEP", default = None,\
		help = "Additional fields to keep [default: %default]")
parser.add_option("-n","--treelimit", dest = "NTREES", default = 0,\
		help = "Number of trees to include in prediction [default: %default]", type = "int")
parser.add_option("-d","--delim", dest = "DELIMITER", default = "|",\
		help = "Input file delimiter [default: %default]", type = "string")
options,args = parser.parse_args()

if options.KEEP:
	KEEP = parseFields(options.KEEP)
else:
	KEEP = None

if options.NTREES != 0:
	NTREES = options.NTREES - 1
else:
	NTREES = options.NTREES

MODEL_FILE = args[0]

with(open(MODEL_FILE,'r')) as f:
	modelDict = pickle.load(f)

model = modelDict['model']
# per XGBoost documentation, predict function is not thread save
# Manually set nthreads to 1 to override value provided during training
model.set_param({"nthread":1})
features = modelDict['variables']
missing = float(modelDict['configuration']['options']['options']['MISSING'])
reader = csv.DictReader(sys.stdin, delimiter = options.DELIMITER)

for i,line in enumerate(reader):
	if i == 0:
		header = reader.fieldnames
		keep_vars = []
		if KEEP:
			keep_vars = [header[i] for i in KEEP]
		header = [x for x in header if x in features]
		assert(len(set(header)) == len(set(features)))
		writer = csv.DictWriter(sys.stdout, delimiter = "|", fieldnames = keep_vars + ['pred0','pred1'])
		writer.writeheader()
	X = np.array([line[var] for var in features])
	X = X.reshape(1, X.shape[0])
	dtest = xgb.DMatrix(X, feature_names = features, missing = missing)
	score = np.asscalar(model.predict(dtest, ntree_limit = NTREES))
	output = {}
	if KEEP:
		output = dict((k,line[k]) for k in keep_vars)
	output['pred0'] = 1. - score
	output['pred1'] = score
	writer.writerow(output)

