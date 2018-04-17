#!/usr/bin/env python
import numpy

def calcKS(tag, score, weights = None):
	if weights is None:
		weights = numpy.ones(len(score))

	score = numpy.array(score)
	tag = numpy.array(tag)
	weights = numpy.array(weights)

	index = numpy.argsort(score)[::-1]
	zeroSoFar = 0.
	oneSoFar = 0.
	maxKS = 0.
	lastScore = float('nan')
	totalZero = numpy.sum(weights[tag == 0])
	totalOne = numpy.sum(weights[tag == 1])

	for ind in index:
		thisScore = score[ind]
		thisWeight = weights[ind]
		if thisScore != lastScore:
			lastScore = thisScore
			ks = (oneSoFar/totalOne)-(zeroSoFar/totalZero)
			if abs(ks)>abs(maxKS):
				maxKS=ks
		if tag[ind] == 0:
			zeroSoFar+=thisWeight
		elif tag[ind] ==1:
			oneSoFar+=thisWeight
	return maxKS

def calcKS_Dmat(tag, score, weights = None):
	if weights is None:
		weights = numpy.ones(score.num_row())

	score = numpy.array(score.get_label())
	tag = numpy.array(tag)
	weights = numpy.array(weights)

	index = numpy.argsort(score)[::-1]
	zeroSoFar = 0.
	oneSoFar = 0.
	maxKS = 0.
	lastScore = float('nan')
	totalZero = numpy.sum(weights[tag == 0])
	totalOne = numpy.sum(weights[tag == 1])

	for ind in index:
		thisScore = score[ind]
		thisWeight = weights[ind]
		if thisScore != lastScore:
			lastScore = thisScore
			ks = (oneSoFar/totalOne)-(zeroSoFar/totalZero)
			if abs(ks)>abs(maxKS):
				maxKS=ks
		if tag[ind] == 0:
			zeroSoFar+=thisWeight
		elif tag[ind] ==1:
			oneSoFar+=thisWeight
