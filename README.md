##UnJPEG: removing jpeg atefacts from screenshots with a neural network

Note: this is just a pet project to teach myself about neural networks and Theano, not expecting usable results, but I'll make a lot of noise if it ever works properly.

Note 2: the image data I used to train the ANN is not on github for size reasons.

Log:
2016/09/01	
* Borrowed multi-layer-perceptron (artificial neural network) code from deeplearning.net  
http://deeplearning.net/tutorial/mlp.html  
* Tweaked to change output from a single integer (classification) to the same size as the input (8x8x3 blocks, the same size jpeg uses)

Plans:
* Look into denoising literature to see if there is a better error measure that better preserves high frequencies than MSE
* Try the layer/pooling technique from a CNN

This branch is probably abandoned... the idea was to try running several filters in 'parallel' that would enable the NN to 'pick' which one to apply based on the image. Never really worked; it was either just using one 'path' or all of the weights were converging on the same thing -- never really investigated.

Having checked the denoising paper that used the MLP, I'm going to go back to the basic model and just throw more layers/neurons at it! Maybe normalise the data too.