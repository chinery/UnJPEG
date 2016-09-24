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

import sys
sys.path.append('../training/')
from mlp import MLP, TopLayer, HiddenLayer, unjpeg, rgb2ycbcr, ycbcr2rgb

if __name__ == '__main__':
	# load the saved model
	with open('best_model.pkl', 'rb') as f:
		model = pickle.load(f)
		
	if theano.config.device.startswith("gpu"):
		classifier = model
	else:
		classifier = MLP.existingclassifier(model)	
		
	with open('params.pkl', 'rb') as file:
		params = pickle.load(file)
		blocksize = pickle.load(file)
		
		
	rgbim = scipy.misc.imread("test.jpg",mode='RGB')/255
	im = rgb2ycbcr(rgbim)

	cleanim = ycbcr2rgb(unjpeg(im,classifier,params,blocksize,0.58,0.38))

	res = Image.fromarray(numpy.uint8(numpy.round(cleanim*255)),mode='RGB')
	res.save('test.png')