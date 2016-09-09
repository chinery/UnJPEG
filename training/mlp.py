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
                # rng.uniform(
                #     low=-numpy.sqrt(6. / (n_in + n_out)),
                #     high=numpy.sqrt(6. / (n_in + n_out)),
                #     size=(n_in, n_out)
                # ),
                rng.uniform(
                    low=-numpy.sqrt(1. / n_in),
                    high=numpy.sqrt(1. / n_in),
                    size=(n_in, n_out)
                ),
                # rng.normal(
                #     loc=0,
                #     # scale=numpy.sqrt(6./(n_in+n_out)),
                #     scale=numpy.sqrt(1./n_in),
                #     size=(n_in, n_out)
                # ),
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

    def __init__(self, rng, input, n_in, n_hidden, n_out, h_layers, activation=T.tanh):
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

        self.hiddenLayers = [None]*(h_layers+1)
        self.hiddenLayers[0] = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=activation
        )
        if h_layers > 1:
            for i in range(1,h_layers):
                self.hiddenLayers[i] = HiddenLayer(
                    rng=rng,
                    input=self.hiddenLayers[i-1].output,
                    n_in=n_hidden,
                    n_out=n_hidden,
                    activation=activation
                )
        self.hiddenLayers[h_layers] = HiddenLayer(
                rng=rng,
                input=self.hiddenLayers[h_layers-1].output,
                n_in=n_hidden,
                n_out=n_out,
                activation=activation
            )

        # self.hiddenLayer2 = HiddenLayer(
        #     rng=rng,
        #     input=self.hiddenLayer.output,
        #     n_in=n_hidden,
        #     n_out=n_hidden,
        #     activation=T.tanh
        # )

        self.topLayer = TopLayer(
            input=self.hiddenLayers[h_layers].output
        )
        # end-snippet-2 start-snippet-3
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            T.sum([abs(layer.W).sum() for layer in self.hiddenLayers])
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            T.sum([(layer.W ** 2).sum() for layer in self.hiddenLayers])
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
        self.params = [param for layer in self.hiddenLayers for param in layer.params]
        # end-snippet-3

        # keep track of model input
        self.input = input

        self.output = self.topLayer.output


def load_data(dataset,m=0,s=1):
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

    training_gt = (training_gt-m)/s
    training_in = (training_in-m)/s
    testing_gt = (testing_gt-m)/s
    testing_in = (testing_in-m)/s

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

    # train_set_x, train_set_y = shared_dataset(training_in, training_gt)
    # valid_set_x, valid_set_y = shared_dataset(testing_in, testing_gt)

    train_set_x, train_set_y, valid_set_x, valid_set_y = training_in, training_gt, testing_in, testing_gt

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y)]
    return rval

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


def findGPUlimit(data_x,data_y,min_n=1):
    n = min_n
    total = data_x.shape[0]
    loop = True
    while loop:
        try:
            dataset = shared_dataset(data_x[0:total//n,:],data_y[0:total//n,:])
            loop = False
        except MemoryError as e:
            n = n + 1

    return (dataset,n)


def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='', batch_size=10000, n_hidden=500, classifier=None,h_layers=1,m=0,s=1):
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
    datasets = load_data(dataset,m,s)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    # test_set_x, test_set_y = datasets[2]

    num_of_samples = train_set_x.shape[0]
    

    # compute number of minibatches for training, validation and testing
    # n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    # n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size

    # no need for these to be the same, for validation want as big as will fit on gpu
    vbatch_size = num_of_samples//100

    # n_train_batches = train_set_x.shape[0] // batch_size
    n_valid_batches = valid_set_x.shape[0] // vbatch_size

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

    n_in = 8*8*3

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=n_in,
        n_hidden=n_hidden,
        n_out=n_in,
        h_layers=h_layers,
        activation=lambda x: 1.7159*T.tanh((2/3)*x)
    )

    neurons_per_layer = [n_in] + [n_hidden]*h_layers + [n_in]
    p_per_layer = (0,1) # a number for each parameters per layer, 2: W and b
    neurons_per_layer = [val for val in neurons_per_layer for _ in (0,1)]

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
    (dataset,n_partitions) = findGPUlimit(train_set_x,train_set_y,3)
    shared_x, shared_y = dataset

    batch_x = T.matrix()
    batch_y = T.matrix()

    # validate_model = theano.function(
    #     inputs=[index],
    #     outputs=classifier.error(y),
    #     givens={
    #         x: valid_set_x[index * batch_size:(index + 1) * batch_size],
    #         y: valid_set_y[index * batch_size:(index + 1) * batch_size]
    #     }
    # )
    validate_model = theano.function(
        inputs=[batch_x,batch_y],
        outputs=classifier.error(y),
        givens={
            x: batch_x,
            y: batch_y
        },
        allow_input_downcast=True
    )

    # start-snippet-5
    # compute the gradient of cost with respect to theta (sorted in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - (learning_rate/div) * gparam)
        for param, gparam, div in zip(classifier.params, gparams, neurons_per_layer)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    # train_model = theano.function(
    #     inputs=[index],
    #     outputs=cost,
    #     updates=updates,
    #     givens={
    #         x: train_set_x[index * batch_size: (index + 1) * batch_size],
    #         y: train_set_y[index * batch_size: (index + 1) * batch_size]
    #     }
    # )
    # train_model = theano.function(
    #     inputs=[batch_x,batch_y],
    #     outputs=cost,
    #     updates=updates,
    #     givens={
    #         x: batch_x,
    #         y: batch_y
    #     },
    #     allow_input_downcast=True
    # )
    allindex = T.ivector()
    train_model = theano.function(
        inputs=[allindex],
        outputs=cost,
        updates=updates,
        givens={
            x: shared_x[allindex,:],
            y: shared_y[allindex,:]
        },
        allow_input_downcast=True
    )
    # end-snippet-5

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    partition_size = num_of_samples//n_partitions
    n_train_batches = partition_size//batch_size

    # early-stopping parameters
    patience = 50*n_train_batches  # look as this many examples regardless
    patience_increase = 2*n_train_batches  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.005  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches*n_partitions, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch
                                  
    best_validation_loss = numpy.inf
    itr = 0
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    #####
    # Add image output
    #####
    im = scipy.misc.imread("test.jpg",mode='YCbCr')/255

    prob = numpy.ones(num_of_samples)/num_of_samples
    p_weight = 2

    """
    Can't fit all data on GPU, sending small batches too slow, sending big batches not helping results
    
    So split data into n parts, determined by how much can fit
    Take a sample of total/n examples, weighted by a probability vector
    Put on GPU
    Run training sequentially, updating corresponding entries in probability vector
    When it reaches the end, take another total/n samples
    Repeat n times
    Call this one epoch (not guaranteed to hit all examples)

    """
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for partition_num in range(n_partitions):
            prob = prob/numpy.sum(prob)

            kth = num_of_samples-partition_size
            partition_sample = numpy.argpartition(prob,kth)[kth:]
            prob_partition = prob[partition_sample]
            ppsum = numpy.sum(prob_partition)

            # partition_sample = numpy.random.choice(num_of_samples,partition_size,False,prob)

            shared_x.set_value(numpy.asarray(train_set_x[partition_sample,:],dtype=theano.config.floatX))
            shared_y.set_value(numpy.asarray(train_set_y[partition_sample,:],dtype=theano.config.floatX))

            for index in range(n_train_batches):
                prob_partition= prob_partition/numpy.sum(prob_partition)
                innersample = numpy.random.choice(partition_size,batch_size,False,prob_partition)

                minibatch_avg_cost = train_model(innersample)

                # update probabilities
                # want to incentivise picking different samples, but if the error is higher then
                # return to this sample sooner
                prob_partition[innersample] = numpy.clip(prob_partition[innersample]*p_weight*minibatch_avg_cost,0.0001,1)
                prob[partition_sample[innersample]] = numpy.clip(prob[partition_sample[innersample]]*p_weight*minibatch_avg_cost,0.0001,1)
                # prob[partition_sample[index*batch_size:(index+1)*batch_size]] = prob[partition_sample[index*batch_size:(index+1)*batch_size]]*p_weight*minibatch_avg_cost

                # iteration number
                # itr = (epoch - 1) * n_train_batches*n_partitions + partition_num + (index*batch_size) 
                itr = itr + 1

                workdone = ((itr + 1) % validation_frequency)/validation_frequency
                if(itr%100 == 0):
                    print("\rProgress: [{0:10s}] {1:.1f}% epoch: {2} partition: {3}/{4} batch:{5}/{6} ".format('#' * int(workdone * 10), workdone*100, epoch, partition_num+1, n_partitions, index+1, n_train_batches), end="", flush=True)

                if (itr + 1) % validation_frequency == 0:
                    print("")
                    # compute zero-one loss on validation set
                    # validation_losses = [validate_model(i) for i
                    #                      in range(n_valid_batches)]
                    validation_losses = [validate_model(valid_set_x[i * vbatch_size: (i + 1) * vbatch_size],
                                                        valid_set_y[i * vbatch_size: (i + 1) * vbatch_size]) for i
                                         in range(n_valid_batches)]                     
                    this_validation_loss = numpy.mean(validation_losses)

                    print(
                        'epoch %i, minibatch %i/%i, partition: %i/%i, validation error %f' %
                        (
                            epoch,
                            index + 1,
                            n_train_batches,
                            partition_num,
                            n_partitions,
                            this_validation_loss
                        )
                    )


                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        ## Save results and images
                        with open('results/models/model{:04d}.pkl'.format(epoch), 'wb') as f:
                                pickle.dump(classifier, f)

                        cleanim = unjpeg(im,classifier,m,s)
                        res = Image.fromarray(numpy.uint8(cleanim*255),mode='YCbCr').convert('RGB')
                        res.save('results/outimages/model{:04d}.png'.format(epoch))

                        #improve patience if loss improvement is good enough
                        if (
                            this_validation_loss < best_validation_loss *
                            (1-improvement_threshold)
                        ):
                            patience = max(patience, itr + patience_increase)

                        # test it on the test set
                        # test_losses = [test_model(i) for i
                        #                in range(n_test_batches)]
                        # test_score = numpy.mean(test_losses)

                        print(('     epoch %i, best error %f, improvement %f') %
                              (epoch, this_validation_loss, this_validation_loss-best_validation_loss))

                        best_validation_loss = this_validation_loss
                        best_iter = itr

                        with open('best_model.pkl', 'wb') as f:
                            pickle.dump(classifier, f)

                if patience <= itr:
                    done_looping = True
                    break


    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)


def unjpeg(im,classifier,m=0,s=1):
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

            block = (block-m)/s

            x_data = block.reshape((1,8*8*3))

            y_data = predict_model(x_data)

            nblock = y_data.reshape((8,8,3))

            nblock = (nblock*s)+m

            result[i*8:(i+1)*8,j*8:(j+1)*8,:] = nblock

    return result[0:h,0:w,:]

if __name__ == '__main__':
    test_mlp(n_epochs=1000, batch_size=100,learning_rate=20,n_hidden=2047,h_layers=4,L2_reg=0.0000,m=0.58,s=0.28)
