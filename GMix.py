#~!usr/bin/env python

import pickle as pkl
import pandas as pd
from argparse import ArgumentParser
from parseFields import parseFields
from sklearn.mixture import BayesianGaussianMixture

usage = 'GMix.py [options] training_data'

parser = ArgumentParser('A command line utility for Bayesian Gaussian Mixture Models')
parser.add_argument('train')
parser.add_argument('-o','--output',type=str,default=None,help='Save final model to pkl [default: None]')
parser.add_argument('-x','--x_vars',type=str,default=None,help='X variables to cluster [Linux-style parsing]')
parser.add_argument('-n','--n_components',type=int,default=1,help='number of Dirichlet mixture components [default: 1]')
parser.add_argument('-d','--delimiter',type=str,default="|",help='File delimiter [default: "|"]')
parser.add_argument('-s','--seed',type=int,default=2813308004,help='Random seed [default: 2813308004]')

args = parser.parse_args()

train = pd.read_csv(args.train,sep=args.delimiter)
x_vars = parseFields(args.x_vars)
data = train[x_vars]
header = list(data)

gmix = BayesianGaussianMixture(n_components=args.n_components,weight_concentration_prior_type='dirichlet_process',\
		init_params='kmeans',max_iter=500,tol=0.001,random_state = args.seed,verbose=2)

gmix.fit(data)

if args.output is not None:
	if args.output.split('.')[-1] != "pkl":
		args.output = args.output + ".pkl"
	print "Saving model to %s" %args.output
	output = {'model': gmix, 'variables':header, 'creation_date':"%c", 'configuration':vars(args)}
	with open(args.output,'w') as m:
		pkl.dump(output,m)
