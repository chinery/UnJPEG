#UnJPEG: removing jpeg atefacts from screenshots with a neural network

Note: this is just a pet project to teach myself about neural networks and 
Theano, not expecting usable results, but I'll make a lot of noise if it 
ever works properly.

Log:
2016/09/01	Borrowed multi-layer-perceptron (artificial neural network) code 
			from deeplearning.net
			http://deeplearning.net/tutorial/mlp.html
			Tweaked to change output from a single integer (classification) to
			the same size as the input (8x8x3 blocks, the same size jpeg uses)

Plans:
	Look into denoising literature to see if there is a better error measure
	that better preserves high frequencies than MSE
	Try filtering training examples to include fewer 'plain-colour' samples
