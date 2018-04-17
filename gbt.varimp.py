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

coefs = model.feature_importances_.tolist()

df_dict = {'coefs':coefs, 'variable':features}
weights = pd.DataFrame(df_dict)
weights['abscoefs'] = np.abs(weights.coefs)
weights = weights.sort(['abscoefs'], ascending = False)
weights.to_csv(sys.stdout, sep = "|", columns = ['variable','coefs'], index = False)
