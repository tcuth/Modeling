#import csv
import sys
from argparse import ArgumentParser
import readline
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
from fdr import fraud_detection_rate
import xgboost as xgb
import pickle
import numpy as np
import collections

parser = ArgumentParser(description = 'A tool to conduct partial sensitivity analysis of an XGBoost model')

parser.add_argument('model',type=str)
parser.add_argument('-y', '--tag', type=int, help = 'tag field index')
parser.add_argument('-d', '--delimiter', type=str, default = '|', help = 'input file delimiter [default: "|"]')
parser.add_argument('-n', '--ntrees', type=int, default = -1, help = 'number of trees to use for scoreing [default: ALL]')
LOGODDS = 'log-odds'
PROB = 'probability'
score_choices = [LOGODDS,PROB]
parser.add_argument('-s', '--score_type', type=str, choices = score_choices, default = score_choices[0], help = 'transformation of score to be output [choices:%s, default: %s]' % (score_choices, score_choices[0]) )
parser.add_argument('-r', '--reverse', type=bool, default=False, help='reverse probabilities [default: False]; not used for score_type = "log-odds"')
parser.add_argument('-f', '--factor', type=float, default = 1, help = 'factor on the log-odds [default: 1]; not used for score_type = "probability"')
parser.add_argument('-o', '--offset', type=float, default = 0, help = 'offset on the log-odds [default: 0]; not used for score_type = "probability"')

args = parser.parse_args()

with(open(args.model, 'r')) as f:
	modelDict = pickle.load(f)

model = modelDict['model']
# per XGBoost documentation, predict function is not thread save
# Manually set nthreads to 1 to override value provided during training
model.set_param({"nthread":1})
features = modelDict['variables']
missing = float(modelDict['configuration']['options']['options']['MISSING'])

#reader = csv.DictReader(sys.stdin, args.delimiter)
X = np.genfromtxt('loads.main.test.csv',delimiter = ',', dtype = None)
header = [x for x in list(X[0]) if x in features]
assert(collections.Counter(header) == collections.Counter(features))

tags = X[1:, args.tag - 1].astype(np.float)
index = np.in1d(X[0],features)
data = X[1:,index].astype(np.float)
header = X[0,index]

def score(dtest, ntrees, missing, factor, offset, output_type):
	if output_type == PROB:
		scores = model.predict(dtest, ntree_limit = ntrees, output_margin = False)
		if args.reverse:
			scores = 1-scores
	elif output_type == LOGODDS:
		scores = model.predict(dtest, ntree_limit = ntrees, output_margin = True)
		scores = -1*scores*factor + offset
	return scores

def five_number(scores):
	return (np.min(scores),np.max(scores),np.std(scores),np.mean(scores),np.median(scores))

creditR = importr("creditR")
numpy2ri.activate()
def absKS(preds, y):
	ks = creditR.ksC(preds,y)
	return -1.0 * ks[0]

# FDR function
def negFDR(preds, y):
	fdr = fraud_detection_rate(y, preds, None, 95)[0]
	return -1 * fdr

dtest = xgb.DMatrix(data, label = tags, feature_names = list(header), missing = missing)
scores = score(dtest, args.ntrees, missing, args.factor, args.offset, args.score_type)
minimum,maximum,std,mean,median = five_number(scores-scores)
ks = absKS(scores, tags)
fdr = negFDR(scores, tags)
print '|'.join(["variable","minimum_abs_difference","maximum_abs_difference","std_abs_difference","mean_abs_difference","median_abs_difference","ks","fdr"])
print '|'.join(["OBSERVED",str(minimum),str(maximum),str(std),str(mean),str(median),str(ks),str(fdr)])

for i in range(data.shape[1]):
	var = header[i]
	cols = data.transpose()
	np.random.shuffle(cols[i])
	mod_data = cols.transpose()
	dtest = xgb.DMatrix(mod_data, label = tags, feature_names = list(header), missing = missing)
	var_scores = score(dtest, args.ntrees, missing, args.factor, args.offset, args.score_type)
	abs_score_diff = abs(scores - var_scores)
	minimum,maximum,std,mean,median = five_number(abs_score_diff)
	ks = absKS(var_scores, tags)
	fdr = negFDR(var_scores, tags)
	print '|'.join([var,str(minimum),str(maximum),str(std),str(mean),str(median),str(ks),str(fdr)])


