#!/usr/bin/env python

import pickle
import pandas as pd
from argparse import ArgumentParser
import sys
import xgboost as xgb

# options parser
## use new pkl object from gbt.py that has all necessary information in a dictionary
usage = "Usage: %prog [options] MODEL_PKL"
parser = ArgumentParser('A command line tool to print variable importances from an XGB pkl model')
parser.add_argument('model')
args = parser.parse_args()

MODEL_FILE = args.model

with(open(MODEL_FILE,'r')) as f:
	modelDict = pickle.load(f)

model = modelDict['model']
features = modelDict['variables']

gains = model.get_score(importance_type = 'gain')
covers = model.get_score(importance_type = 'cover')
weights = model.get_score(importance_type = 'weight')
total = sum(weights.values())
weights = {key: float(value)/total for key, value in weights.items()}

dicts = [gains, covers, weights]
importances = {}
for key in gains.iterkeys():
	importances[key] = tuple(d[key] for d in dicts)

df = pd.DataFrame(data = importances.values(), index = importances.keys())
df.columns = ["gain","cover","weight"]
df = df.sort(columns = "gain", ascending = False)
df.to_csv(sys.stdout, sep = "|", columns = df.columns, index = True, index_label = "variable")
