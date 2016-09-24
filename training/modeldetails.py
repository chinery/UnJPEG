import os
import sys
import timeit

import numpy
import scipy
import math

import theano
import theano.tensor as T

from PIL import Image

import pickle

from mlp import MLP, TopLayer, HiddenLayer, unjpeg


if __name__ == '__main__':
	# load the saved model
	with open('model.pkl', 'rb') as f:
		classifier = pickle.load(f)

	print('Hidden layers: ',len(classifier.hiddenLayers))
	print('Block size: ', numpy.sqrt(classifier.hiddenLayers[0].W.get_value().shape[0]/3))
	print('Neurons: ', classifier.hiddenLayers[0].W.get_value().shape[1])