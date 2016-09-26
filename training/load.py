import os
import io
import numpy
import scipy, scipy.misc
import math, random
from PIL import Image
import pickle
import theano
import glob
import matplotlib.pyplot as plt
from mlp import rgb2ycbcr, ycbcr2rgb

def binarysearch(data,search):
	ix = numpy.searchsorted(data,search)
	return ix < len(data) and search == data[ix]

	
#images = [entry for entry in os.scandir('./images/') if entry.is_file()]
images = glob.glob('./images/*.png')

blocksize = 24
partition_size = 20000
ppbulk = 5
maxmemsize = partition_size*ppbulk

offset = ((blocksize//8)-1)
print("Counting blocks in images...")

blocks = 0
for ix,imgpath in enumerate(images):
	im = scipy.misc.imread(imgpath,mode='RGB')
	h,w,c = im.shape
	blocks = blocks + ((math.ceil(h/8)-offset) * (math.ceil(w/8)-offset))

	workdone = ix/len(images)
	print("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(workdone * 50), workdone*100), end="", flush=True)

print("")
print("Total of {} blocks".format(blocks))

testnum = numpy.min((round(blocks*0.25),500000))

testing_gt = numpy.zeros((testnum,blocksize*blocksize*3),dtype=theano.config.floatX)
testing_in = numpy.zeros((testnum,blocksize*blocksize*3),dtype=theano.config.floatX)

bulktraining_gt = numpy.zeros((maxmemsize,blocksize*blocksize*3),dtype=theano.config.floatX)
bulktraining_in = numpy.zeros((maxmemsize,blocksize*blocksize*3),dtype=theano.config.floatX)

trainorder = numpy.arange(0,maxmemsize)
numpy.random.shuffle(trainorder)

testchoice = numpy.random.choice(blocks,testnum,False)
testchoice.sort()

testorder = numpy.arange(0,testnum)
numpy.random.shuffle(testorder)

#lastpartition = (blocks-testnum) // partition_size
#lpsize = blocks - testnum - (lastpartition*partition_size)

lastbulk = (blocks-testnum) // maxmemsize
lbsize = blocks - testnum - (lastbulk*maxmemsize)

lastppbulk = (lbsize // partition_size) + 1

print("Computing JPEGs and storing blocks")
overall_count = 0
train_count = 0
test_count = 0
partition_count = 0
bulk_count = 0
for ix,imgpath in enumerate(images):
	rgbim = scipy.misc.imread(imgpath,mode='RGB')/255
	h,w,c = rgbim.shape
	
	if w%8 != 0 or h%8 != 0:
		rgbnewim = numpy.zeros((math.ceil(h/8)*8, math.ceil(w/8)*8,3))
		rgbnewim[0:h,0:w,:] = rgbim
		rgbnewim[h:math.ceil(h/8)*8,0:w,:] = rgbim[h-1:h,0:w,:]
		rgbnewim[0:h,w:math.ceil(w/8)*8,:] = rgbim[0:h,w-1:w,:]
		rgbnewim[h:math.ceil(h/8)*8,w:math.ceil(w/8)*8,:] = rgbim[h-1:h,w-1:w,:]
	else:
		rgbnewim = rgbim

	pilim = Image.fromarray(numpy.uint8(rgbnewim*255),mode='RGB')
	
	newim = rgb2ycbcr(rgbnewim)
	
	inmem = io.BytesIO()
	# q = random.randint(25,75)
	# ss = random.randint(0,2)
	# blockim.save(inmem,format='jpeg',quality=q,subsampling=ss)
	# these are twitter's settings
	pilim.save(inmem,format='jpeg',quality=85,subsampling=2,progressive=1)
	rgbjpegim = scipy.misc.imread(inmem,mode='RGB')/255
	
	jpegim = rgb2ycbcr(rgbjpegim)

	for i in range(0,math.ceil(h/8)-offset):
		for j in range(0,math.ceil(w/8)-offset):
			block = newim[i*8:(i*8)+blocksize,j*8:(j*8)+blocksize,:]
			jpegblock = jpegim[i*8:(i*8)+blocksize,j*8:(j*8)+blocksize,:]

			if not binarysearch(testchoice,overall_count):
				bulktraining_gt[trainorder[train_count],:] = block.reshape((1,blocksize*blocksize*3))
				bulktraining_in[trainorder[train_count],:] = jpegblock.reshape((1,blocksize*blocksize*3))
				train_count += 1
				if train_count == maxmemsize:
					# save a batch of partitions
					print('\nSaving partitions {} to {}'.format(partition_count,partition_count+ppbulk-1))
					for partix in range(0,ppbulk):
						training_gt = bulktraining_gt[partix*partition_size:(partix+1)*partition_size,:]
						training_in = bulktraining_in[partix*partition_size:(partix+1)*partition_size,:]
						with open('partitioneddata/partition{:06d}.pkl'.format(partition_count+partix), 'wb') as file:
							numpy.savez_compressed(file,training_gt=training_gt,training_in=training_in)
					partition_count += ppbulk
					
					bulk_count += 1
					if bulk_count == lastbulk:
						maxmemsize = lbsize
						bulktraining_gt = numpy.zeros((maxmemsize,blocksize*blocksize*3),dtype=theano.config.floatX)
						bulktraining_in = numpy.zeros((maxmemsize,blocksize*blocksize*3),dtype=theano.config.floatX)
						trainorder = numpy.arange(0,maxmemsize)
						ppbulk = lastppbulk
					elif bulk_count > lastbulk:
						print('reached last stuff!')
					numpy.random.shuffle(trainorder)
					train_count = 0
			else:
				testing_gt[testorder[test_count],:] = block.reshape((1,blocksize*blocksize*3))
				testing_in[testorder[test_count],:] = jpegblock.reshape((1,blocksize*blocksize*3))
				test_count += 1

			overall_count += 1

	workdone = ix/len(images)
	print("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(workdone * 50), workdone*100), end="", flush=True)


print("")
print("{} testing blocks".format(testnum))


print("Saving test data")

with open('testdata.pkl', 'wb') as file:
	numpy.savez_compressed(file,testing_gt=testing_gt,testing_in=testing_in)

