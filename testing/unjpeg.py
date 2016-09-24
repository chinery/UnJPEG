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
from mlp import MLP, TopLayer, HiddenLayer, unjpeg

if __name__ == '__main__':
	# load the saved model
	with open('best_model.pkl', 'rb') as f:
		classifier = pickle.load(f)

	with open('params.pkl', 'rb') as file:
		params = pickle.load(file)
		blocksize = pickle.load(file)
		
	# im = scipy.misc.imread("test2.jpg",mode='YCbCr')/255
	im = scipy.misc.imread("differentscreen.jpg",mode='YCbCr')/255

	cleanim = unjpeg(im,classifier,params,blocksize,0.58,0.38)

	res = Image.fromarray(numpy.uint8(cleanim*255),mode='YCbCr').convert('RGB')
	res.save('nontwitter.png')