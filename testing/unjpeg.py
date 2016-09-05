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
from mlp import MLP, TopLayer, HiddenLayer, PoolLayer, unjpeg


# def unjpeg(im):
#     """
#     Split the image into blocks, run through the neural net
#     """

#     # compile a predictor function
#     predict_model = theano.function(
#         inputs=[classifier.input],
#         outputs=classifier.output,allow_input_downcast=True)


#     h,w,c = im.shape
#     if w%8 != 0 or h%8 != 0:
#         newim = numpy.zeros((math.ceil(h/8)*8, math.ceil(w/8)*8,3))
#         newim[0:h,0:w,:] = im
#         newim[h:math.ceil(h/8)*8,0:w,:] = im[h-1:h,0:w,:]
#         newim[0:h,w:math.ceil(w/8)*8,:] = im[0:h,w-1:w,:]
#         newim[h:math.ceil(h/8)*8,w:math.ceil(w/8)*8,:] = im[h-1:h,w-1:w,:]
#     else:
#         newim = im

#     result = numpy.zeros((math.ceil(h/8)*8, math.ceil(w/8)*8,3))

#     for i in range(0,math.ceil(h/8)):
#         for j in range(0,math.ceil(w/8)):
#             block = newim[i*8:(i+1)*8,j*8:(j+1)*8,:]
            
#             x_data = block.reshape((1,8*8*3))

#             y_data = predict_model(x_data)

#             result[i*8:(i+1)*8,j*8:(j+1)*8,:] = y_data.reshape((8,8,3))

#     return result[0:h,0:w,:]

if __name__ == '__main__':
    # load the saved model
    with open('best_model.pkl', 'rb') as f:
        classifier = pickle.load(f)

    im = scipy.misc.imread("test.jpg",mode='YCbCr')/255

    cleanim = unjpeg(im,classifier)

    res = Image.fromarray(numpy.uint8(cleanim*255),mode='YCbCr').convert('RGB')
    res.save('result.png')