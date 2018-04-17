#!/data/bin/anaconda/bin/python

import pickle
import csv
import pandas as pd
import numpy as np
from optparse import OptionParser
import sys
from sklearn.ensemble import GradientBoostingClassifier
from load_data import load_data
sys.path.append('/data/bin/')
from parseFields import parseFields
# options parser
## use new pkl object from gbt.py that has all necessary information in a dictionary
usage = "Usage: cat INPUT_FILE | gbt.predict.py [options] MODEL_PKL > OUTPUT_FILE"

parser = OptionParser(usage = usage)
parser.add_option("-k","--keep", dest = "KEEP", default = None,\
		help = "Additional fields to keep [default: %default]")
options,args = parser.parse_args()

if options.KEEP:
	KEEP = parseFields(options.KEEP)
else:
	KEEP = None

MODEL_FILE = args[0]

with(open(MODEL_FILE,'r')) as f:
	modelDict = pickle.load(f)

model = modelDict['model']
features = modelDict['variables']

reader = csv.DictReader(sys.stdin, delimiter = "|")

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
	score = model.predict_proba(X)
	output = {}
	if KEEP:
		output = dict((k,line[k]) for k in keep_vars)
	output['pred0'] = score[0,0]
	output['pred1'] = score[0,1]
	writer.writerow(output)

