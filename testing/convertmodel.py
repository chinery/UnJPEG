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
        raise ValueError('Set device to cpu to produce cpu model')
        
    with open('gpu_model.pkl', 'rb') as f:
        classifier = pickle.load(f)
    
    cpu_model = MLP.existingclassifier(classifier)
    
    with open('cpu_model.pkl', 'wb') as f:
        pickle.dump(cpu_model,f)
    