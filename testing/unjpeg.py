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
    if theano.config.device.startswith("gpu"):
        with open('gpu_model.pkl', 'rb') as f:
            classifier = pickle.load(f)
    else:
        with open('cpu_model.pkl', 'rb') as f:
            classifier = pickle.load(f)
        
    blocksize = numpy.uint32(numpy.sqrt(classifier.hiddenLayers[0].W.get_value().shape[0]/3))
    rgbim = scipy.misc.imread("test.jpg",mode='RGB')/255
    im = rgb2ycbcr(rgbim)

    cleanim = ycbcr2rgb(unjpeg(im,classifier,blocksize,0.58,0.38))

    res = Image.fromarray(numpy.uint8(numpy.round(cleanim*255)),mode='RGB')
    res.save('test.png')