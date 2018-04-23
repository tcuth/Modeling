#!/usr/bin/env python

import pickle
import csv
import pandas as pd
import numpy as np
from argparse import ArgumentParser
import sys
import xgboost as xgb
from load_data import load_data
sys.path.append('/data/bin/')
from parseFields import parseFields
# arguments parser
## use new pkl object from xgb.py that has all necessary information in a dictionary
usage = "Usage: cat INPUT_FILE | gbt.predict.py [options] MODEL_PKL > OUTPUT_FILE"

parser = ArgumentParser(usage = usage)
parser.add_argument("model_file")
parser.add_argument("-k","--keep", default = None,\
		help = "Additional fields to keep [default: None]")
parser.add_argument("-n","--treelimit", default = 0,\
		help = "Number of trees to include in prediction [default: 0]", type = int)
parser.add_argument("-d","--delimiter", default = "|",\
		help = 'Input file delimiter [default: "|"]', type = str)
args = parser.parse_args()

if args.keep:
	KEEP = parseFields(args.keep)
else:
	KEEP = None

if args.treelimit != 0:
	NTREES = args.treelimit - 1
else:
	NTREES = args.treelimit

MODEL_FILE = args.model_file

with(open(MODEL_FILE,'r')) as f:
	modelDict = pickle.load(f)

model = modelDict['model']
# per XGBoost documentation, predict function is not thread save
# Manually set nthreads to 1 to override value provided during training
model.set_param({"nthread":1})
features = modelDict['variables']
missing = float(modelDict['configuration']['options']['options']['missing'])
reader = csv.DictReader(sys.stdin, delimiter = args.delimiter)

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
	score = np.asscalar(model.predict(dtest, ntree_limit = args.treelimit))
	output = {}
	if KEEP:
		output = dict((k,line[k]) for k in keep_vars)
	output['pred0'] = 1. - score
	output['pred1'] = score
	writer.writerow(output)

