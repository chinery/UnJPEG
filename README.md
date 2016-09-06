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

Ideas:
8*8 blocks instead of 8*8*3
	Train over the Y, Cb, Cr channels separately. I THINK (check) that after chroma ss, jpeg quantises them using the same table, so it should work. This means more training data for free, and a smaller input layer. It also means being able to skip the chroma ss step... can train on non-ss images: would take more work on the denoiser (for images with ss on) but should be easy enough to just sample the chroma layers differently to create the 8x8 blocks
Choose batches according to error
	Create a probability vector over all training samples
	Rather than going through each block of x samples at once, form the block of x samples according to the probability vector
	Update the probability to make (a) all other samples more likely (to encourage variation) and (b) weight the probabilities for the ones chosen proportionally to the error produced: lower error samples will be less likely to be picked
	The role of the epoch is less meaningful: some samples may fall out of favour early on (the error was very low) and not get used again for a long time