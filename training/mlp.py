"""
Code borrowed and modified from the Theano tutorial
Copyright (c) 2008–2013, Theano Development Team All rights reserved.

Redistribution and use in source and binary forms, with or without modification
, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this 
list of conditions and the following disclaimer. Redistributions in binary form
must reproduce the above copyright notice, this list of conditions and the 
following disclaimer in the documentation and/or other materials provided with 
the distribution.

Neither the name of Theano nor the names of its contributors may be used to 
endorse or promote products derived from this software without specific prior 
written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ‘’AS IS’’ AND ANY EXPRESS 
OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES 
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY DIRECT, INDIRECT, 
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR 
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF 
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Original file + modifications follow:

This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""

from __future__ import print_function

__docformat__ = 'restructedtext en'


import os
import sys
import timeit

import numpy
import scipy
import math
from PIL import Image

import theano
import theano.tensor as T
from theano import pp
import theano.tensor.basic
from theano.tensor import fft

import pickle


# start-snippet-1
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]

# chin
class TopLayer(object):
    """ Layer which just contains error functions
        Has no maths, so the output will be the same dimensionality as the input
        So make sure whatever is being passed in matches the dimensionality of the ground truth
    """
    def __init__(self, input):
        self.input = input
        self.output = input


    # def negative_log_likelihood(self, y):
    #     """Return the mean of the negative log-likelihood of the prediction
    #     of this model under a given target distribution.

    #     .. math::

    #         \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
    #         \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
    #             \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
    #         \ell (\theta=\{W,b\}, \mathcal{D})

    #     :type y: theano.tensor.TensorType
    #     :param y: corresponds to a vector that gives for each example the
    #               correct label

    #     Note: we use the mean instead of the sum so that
    #           the learning rate is less dependent on the batch size
    #     """
    #     # start-snippet-2
    #     # y.shape[0] is (symbolically) the number of rows in y, i.e.,
    #     # number of examples (call it n) in the minibatch
    #     # T.arange(y.shape[0]) is a symbolic vector which will contain
    #     # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
    #     # Log-Probabilities (call it LP) with one row per example and
    #     # one column per class LP[T.arange(y.shape[0]),y] is a vector
    #     # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
    #     # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
    #     # the mean (across minibatch examples) of the elements in v,
    #     # i.e., the mean log-likelihood across the minibatch.
    #     return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
    #     # end-snippet-2

    def error(self, y):
        """Return a float representing error in the minibatch

        :type y: theano.tensor.TensorType
        :param y: matrix which gives correct unjpeg
        """

        # check if y has same dimension of output
        if y.ndim != self.output.ndim:
            raise TypeError(
                'y should have the same shape as self.output',
                ('y', y.type, 'output', self.output.type)
            )
        
        return T.sqrt(T.mean((self.output - y)**2))
        # return T.sqrt(T.mean((fft.rfft(self.output)-fft.rfft(y))**2))
        
class ScoreLayer(HiddenLayer):
    """ Take input from n-dimensions and output a float
    """

    def __init__(self,rng, input, n_in, n_out, W=None, b=None):
        HiddenLayer.__init__(self,rng,input,n_in,n_out,W,b,T.mean)


class PoolLayer(object):
    """ Take n inputs and n scores and output the one with the best score
    """
    def __init__(self,imageinput,scoreinput):
        imgs = T.stack(imageinput)
        # label = T.zeros_like(T.stack(scoreinput))
        # label = T.set_subtensor(label[T.argmax(scoreinput)],1)
        # self.output = T.dot(label,imgs)
        self.output = imgs[T.argmax(scoreinput)]



# start-snippet-2
class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out, paralayers):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        self.hiddenLayers = [None]*paralayers
        self.scoreLayers = [None]*paralayers
        for i in range(0,paralayers):
            self.hiddenLayers[i] = HiddenLayer(
                rng=numpy.random.RandomState(1000+(i*2)),
                input=input,
                n_in=n_in,
                n_out=n_hidden,
                activation=T.tanh
            )
            if i == 0:
                self.scoreLayers[i] = ScoreLayer(
                    rng=numpy.random.RandomState(1001+(i*2)),
                    input=self.hiddenLayers[i].output,
                    n_in=n_in,
                    n_out=n_hidden
                )
            else:
                self.scoreLayers[i] = ScoreLayer(
                    rng=numpy.random.RandomState(1001+(i*2)),
                    input=self.hiddenLayers[i].output,
                    n_in=n_in,
                    n_out=n_hidden,
                    W=self.scoreLayers[0].W,
                    b=self.scoreLayers[0].b
                )


        self.poolLayer = PoolLayer(
            imageinput=[layer.output for layer in self.hiddenLayers],
            scoreinput=[layer.output for layer in self.scoreLayers]
        )


        self.topLayer = TopLayer(
            input=self.poolLayer.output
        )
        # end-snippet-2 start-snippet-3
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        # self.L1 = (
        #     abs(self.hiddenLayer.W).sum()
        # )
        self.L1 = (
            T.sum([abs(layer.W).sum() for layer in self.hiddenLayers]) +
            abs(self.scoreLayers[0].W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        # self.L2_sqr = (
        #     (self.hiddenLayer.W ** 2).sum()
        # )
        self.L2_sqr = (
            T.sum([(layer.W ** 2).sum() for layer in self.hiddenLayers]) +
            (self.scoreLayers[0].W**2).sum()
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        # self.negative_log_likelihood = (
        #     self.topLayer.negative_log_likelihood
        # )
        # same holds for the function computing the number of errors
        self.error = self.topLayer.error

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = [param for layer in self.hiddenLayers for param in layer.params] + self.scoreLayers[0].params
        # end-snippet-3

        # keep track of model input
        self.input = input

        self.output = self.topLayer.output


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset 
    '''

    #############
    # LOAD DATA #
    #############
    print('... loading data')

    with numpy.load('data.pickle', 'rb') as data:
        # training_gt = data['arr_0']
        # training_in = data['arr_1']
        # testing_gt = data['arr_2']
        # testing_in = data['arr_3']
        training_gt = data['training_gt']
        training_in = data['training_in']
        testing_gt = data['testing_gt']
        testing_in = data['testing_in']

    # # Load the dataset
    # with gzip.open(dataset, 'rb') as f:
    #     try:
    #         train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    #     except:
    #         train_set, valid_set, test_set = pickle.load(f)
    # # train_set, valid_set, test_set format: tuple(input, target)
    # # input is a numpy.ndarray of 2 dimensions (a matrix)
    # # where each row corresponds to an example. target is a
    # # numpy.ndarray of 1 dimension (vector) that has the same length as
    # # the number of rows in the input. It should give the target
    # # to the example with the same index in the input.

    def shared_dataset(data_x, data_y, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, shared_y


    train_set_x, train_set_y = shared_dataset(training_in, training_gt)
    valid_set_x, valid_set_y = shared_dataset(testing_in, testing_gt)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y)]
    return rval





def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='', batch_size=10000, n_hidden=500, classifier=None, paralayers=1):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


   """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    # test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    # n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.matrix('y')  # and again

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=8*8*3,
        n_hidden=n_hidden,
        n_out=8*8*3,
        paralayers=paralayers
    )

    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.error(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )
    # end-snippet-4

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    # test_model = theano.function(
    #     inputs=[index],
    #     outputs=classifier.error(y),
    #     givens={
    #         x: test_set_x[index * batch_size:(index + 1) * batch_size],
    #         y: test_set_y[index * batch_size:(index + 1) * batch_size]
    #     }
    # )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.error(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # start-snippet-5
    # compute the gradient of cost with respect to theta (sorted in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param,disconnected_inputs='ignore') for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-5

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2*n_train_batches  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.0005  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    #####
    # Add image output
    #####
    im = scipy.misc.imread("test.jpg",mode='YCbCr')/255


    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss
                    )
                )

                ## Save results and images
                with open('results/models/model{:04d}.pkl'.format(epoch), 'wb') as f:
                        pickle.dump(classifier, f)

                cleanim = unjpeg(im,classifier)
                res = Image.fromarray(numpy.uint8(cleanim*255),mode='YCbCr').convert('RGB')
                res.save('results/outimages/model{:04d}.png'.format(epoch))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        (1-improvement_threshold)
                    ):
                        patience = max(patience, iter + patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    # test_losses = [test_model(i) for i
                    #                in range(n_test_batches)]
                    # test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           best_validation_loss))

                    with open('best_model.pkl', 'wb') as f:
                        pickle.dump(classifier, f)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)


def unjpeg(im,classifier):
    """
    Apply to an image
    """

    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.output,allow_input_downcast=True)


    h,w,c = im.shape
    if w%8 != 0 or h%8 != 0:
        newim = numpy.zeros((math.ceil(h/8)*8, math.ceil(w/8)*8,3))
        newim[0:h,0:w,:] = im
        newim[h:math.ceil(h/8)*8,0:w,:] = im[h-1:h,0:w,:]
        newim[0:h,w:math.ceil(w/8)*8,:] = im[0:h,w-1:w,:]
        newim[h:math.ceil(h/8)*8,w:math.ceil(w/8)*8,:] = im[h-1:h,w-1:w,:]
    else:
        newim = im

    result = numpy.zeros((math.ceil(h/8)*8, math.ceil(w/8)*8,3))

    for i in range(0,math.ceil(h/8)):
        for j in range(0,math.ceil(w/8)):
            block = newim[i*8:(i+1)*8,j*8:(j+1)*8,:]
            
            x_data = block.reshape((1,8*8*3))

            y_data = predict_model(x_data)

            result[i*8:(i+1)*8,j*8:(j+1)*8,:] = y_data.reshape((8,8,3))

    return result[0:h,0:w,:]

if __name__ == '__main__':
    test_mlp(n_epochs=1000, batch_size=1000,learning_rate=0.1,n_hidden=192,L2_reg=0.0001,paralayers=15)
