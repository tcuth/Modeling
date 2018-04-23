#~!usr/bin/env python

import sys
import pickle as pkl
import pandas as pd
from argparse import ArgumentParser
from parseFields import parseFields
from sklearn.mixture import BayesianGaussianMixture

usage = 'cat file | GMix.py [options] training_data > predictions'

parser = ArgumentParser('A command line utility for Bayesian Gaussian Mixture Models', usage = usage)
parser.add_argument('model_file')
parser.add_argument('-d','--delimiter',type=str,default="|",help='File delimiter [default: "|"]')
args = parser.parse_args()


predict = pd.read_csv(sys.stdin, sep = args.delimiter)
with(open(args.model_file,'r')) as f:
	modelDict = pkl.load(f)

x_vars = parseFields(modelDict['configuration']['x_vars'])
predict = predict[x_vars]
gmix = modelDict['model']
preds = gmix.predict_proba(predict)
clusters = preds.argmax(axis=1)
preds = pd.DataFrame(preds)
preds.columns = ["cluster_"+str(col) for col in list(preds)]
predict = pd.concat([predict, preds], axis = 1)
predict['cluster'] = clusters
predict.to_csv(sys.stdout, sep = args.delimiter, index = False)
