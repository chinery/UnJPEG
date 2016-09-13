##UnJPEG: removing jpeg atefacts from screenshots with a neural network

Note: this is just a pet project to teach myself about neural networks and Theano, not expecting usable results, but I'll make a lot of noise if it ever works properly.

Note 2: the image data I used to train the ANN is not on github for size reasons.

Log:
2016/09/13
* Tried a lot of ideas in the past week or so, in no particular order:
* * Shuffle the input blocks
* * Creating multiple 'parallel' layers like a CNN, initally choosing one using a 'predictor' layer (like pooling), later trying a fully combined network. Never really improved performance.
* * I finally read the Burger et al. 2012 paper [1] and some of the book 'Neural networks: Tricks of the trade' [2]. Learned lots of tricks. Tried normalising each image but that didn't work at all, so implemented parameters to normalise the whole dataset. Implemented many suggestions from the book: change learning rate per layer in proportion to incoming connections, normalising, using (1.7...)tanh((2/3)x) as the activation function. Used the network parameters from the paper: 2047 neurons in 4 hidden layers. This required moving to GPU acceleration which itself had some headaches (I set up a system for moving data on and off the GPU in batches to reduce overhead). This seemed to get some promising results.
* * One suggestion from the book was to increase the chance of repeating training examples which had higher errors, because these will best train the network. I spent a while implementing some different forms of this, but I could never really tell how successful it was.
* I'm trying my last few ideas on my Windows PC (more headaches) which has the best GPU. I believe I'm almost at something which you could call a result and I'll roll my changes back into the master branch once I've got that working.
2016/09/01	
* Borrowed multi-layer-perceptron (artificial neural network) code from deeplearning.net  
http://deeplearning.net/tutorial/mlp.html  
* Tweaked to change output from a single integer (classification) to the same size as the input (8x8x3 blocks, the same size jpeg uses)

Plans:
* Look into denoising literature to see if there is a better error measure that better preserves high frequencies than MSE
* Try the layer/pooling technique from a CNN

[1] http://www.hcburger.com/files/neuraldenoising.pdf
[2] http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
