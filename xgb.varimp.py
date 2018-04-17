#!/usr/bin/env python

import pickle
import csv
import pandas as pd
import numpy as np
from optparse import OptionParser
import sys
import operator
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

coefs = model.get_fscore()
coefs = sorted(coefs.items(), key = operator.itemgetter(1))

weights = pd.DataFrame(coefs, columns = ['variable','coefs'])
weights['abscoefs'] = np.abs(weights.coefs)
weights['coefs'] = weights['abscoefs']/weights['abscoefs'].sum()
weights = weights.sort_values(by='coefs', ascending = False)
weights.to_csv(sys.stdout, sep = "|", columns = ['variable','coefs'], index = False)
