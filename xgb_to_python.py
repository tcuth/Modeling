import pickle
import xgboost as xgb

MODEL_FILE = 'xgb.pcs.final.no_rel.pkl'

with(open(MODEL_FILE,'r')) as f:
	modelDict = pickle.load(f)

model = modelDict['model']
MISSING = modelDict['configuration']['options']['options']['MISSING']

def get_code(model, spacer_base="\t"):

	def code_tree(tree, spacer_base, start_depth = 0):

		def node_to_ruledict(node_str):
			is_leaf = False
			k,v = node.split(":")
			try:
				condition_str,action_str = v.split(" ")
				feature,threshold = condition_str.replace("[","").replace("]","").split("<")
				threshold = "float('"+threshold+"')" if threshold == "inf" else threshold
				yes,no,missing = [action.split("=")[1] for action in action_str.split(",")]
				node_attr = {"feature": feature, "threshold": threshold, "yes": yes, "no": no, "missing": missing}
			except ValueError:
				leaf = v.split("=")[1]
				is_leaf = True
				node_attr = {"p": leaf}
			return k, (is_leaf, node_attr)

		def recurse(ruledict, nodepos, depth = 0):
			spacer = spacer_base * depth
			is_leaf, node_attr = ruledict[nodepos]
			if not is_leaf:
				if node_attr["yes"] == node_attr["missing"]:
					missing_condition = " or "
				else:
					missing_condition = " and not "
				print spacer + "if row_dict['" + node_attr["feature"] + "'] < " + node_attr["threshold"] + \
						missing_condition + "row_dict['" + node_attr["feature"] + "'] == MISSING:"
				recurse(ruledict, node_attr["yes"], depth+1)
				print spacer + "else:"
				recurse(ruledict, node_attr["no"], depth+1)
			else:
				print spacer + "logit_bias += " + node_attr["p"]

		nodes = [node.replace('\t','') for node in tree.split('\n')][:-1]

		ruledict = {}
		for node in nodes:
			k,v = node_to_ruledict(node)
			ruledict[k] = v

		recurse(ruledict, "0", start_depth)

	print "import math"
	print
	print "def PCSboost(row_dict, population_logit = 0):"
	print spacer_base + "logit_bias = population_logit"
	print spacer_base + "MISSING = " + str(MISSING)

	tree_list = model.get_dump()
	tree_count = 0
	for tree in tree_list:
		if tree_count < 6500:
			code_tree(tree, spacer_base, start_depth = 1)
			tree_count += 1

	print spacer_base + "odds = math.exp(logit_bias)"
	print spacer_base + "prob = odds / (1 + odds)"
	print spacer_base + "return prob"

_ = get_code(model)
