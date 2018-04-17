#!/home/llockhart/anaconda/bin/python

import pickle
import csv
import pandas as pd
import numpy as np
from optparse import OptionParser
import sys


# options parser
usage = "Usage: %prog [options] MODEL_PKL FEATURE_FILE"
parser = OptionParser(usage = usage)
parser.add_option("-d","--delimiter", dest = "DELIMITER", help = "Feature file delimiter",\
	default = "|")
parser.add_option("-s","--suppress", dest = "SUPPRESS", help = "Columns to ignore", \
	default = "primary_id,isCreditBad,isFraud,client,weight,segment1,segment2,application_date,pred0,pred1,portfolio_type,data_provider")
options,args = parser.parse_args()

MODEL_FILE = args[0]
FEATURE_FILE = args[1]
SUPPRESS = options.SUPPRESS.split(",")
DELIMITER = options.DELIMITER

with(open(MODEL_FILE,'r')) as f:
	model = pickle.load(f)

coefs = model.feature_importances_.tolist()

with(open(FEATURE_FILE,'r')) as f:
	features = f.readline().strip().split(DELIMITER)
	features = [x for x in features if x not in SUPPRESS]

df_dict = {'coefs':coefs, 'variable':features}
weights = pd.DataFrame(df_dict)
weights['abscoefs'] = np.abs(weights.coefs)
weights = weights.sort(['abscoefs'], ascending = False)
weights.to_csv(sys.stdout, sep = "|", columns = ['variable','coefs'], index = False)
