#!/home/llockhart/anaconda/bin/python

import sys
import csv
from sklearn.linear_model import RandomizedLogisticRegression
import pandas as pd
import numpy as np
from optparse import OptionParser
from load_data import load_data

sys.path.append('/home/llockhart/helpers/')
from parseFields import parseFields
from combineFields import combineFields

usage = "random.regression.py [options] FILE"
parser = OptionParser(usage = usage)
parser.add_option("-d","--delimiter",default = "|", dest = "DELIMITER",\
		help = "Input file delimiter [default: %default]")
parser.add_option("-x","--x-vars", default = "1", dest = "X_VARS",\
		help = "Dependent variables [default: %default]")
parser.add_option("-y","--y-vars", default = "2", dest = "Y_VAR",\
		help = "Dependent variables [default: %default]")
parser.add_option("-p","--proportion", default = 0.5, dest = "PROP",\
		help = "Fraction of records to use during each run [default: %default]")
parser.add_option("-n","--n-resampling", default = 200, dest = "NSAMP", \
		help = "Number of randomized models [default: %default]")
parser.add_option("-s","--seed", default = 8675309, dest = "SEED",\
		help = "Random seed [default: %default]")
options,args = parser.parse_args()

DATA = args[0]
DELIMITER = str(options.DELIMITER)
X_VARS = str(options.X_VARS)
Y_VAR = int(options.Y_VAR)
PROP = float(options.PROP)
NSAMP = int(options.NSAMP)
SEED = int(options.SEED)

X, Y, _, dv, header = load_data(DATA,X_VARS,Y_VAR,None)

rlr = RandomizedLogisticRegression(C = [0.01, 0.1, 1.0], sample_fraction = PROP,\
		n_resampling = NSAMP, verbose = 1.0, n_jobs = 1, normalize = True, random_state = SEED)
rlr.fit(X,Y)

varimp = [(var,score) for var,score in zip(header, rlr.scores_) if score > 0.0]
varimp = sorted(varimp, key=lambda tup: tup[1], reverse = True)

print "variable" + "|" + "score"
for var,score in varimp:
	print str(var) + "|" + str(score)

