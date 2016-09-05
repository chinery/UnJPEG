import os
import io
import numpy
import scipy, scipy.misc
import math, random
from PIL import Image
import pickle

images = [entry for entry in os.scandir('./images/') if entry.is_file()]

print("Counting blocks in images...")

blocks = 0
for ix,imgpath in enumerate(images):
	im = scipy.misc.imread(imgpath.path,mode='YCbCr')
	h,w,c = im.shape
	blocks = blocks + (math.ceil(h/8) * math.ceil(w/8))

	workdone = ix/len(images)
	print("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(workdone * 50), workdone*100), end="", flush=True)

print("")
print("Total of {} blocks".format(blocks))

trainix = round(blocks*0.75)

training_gt = numpy.zeros((trainix,8*8*3))
testing_gt = numpy.zeros((blocks-trainix,8*8*3))
training_in = numpy.zeros((trainix,8*8*3))
testing_in = numpy.zeros((blocks-trainix,8*8*3))

print("Computing JPEGs and storing blocks")
count = 0
cskip = 0
final = 0
testing = False
for ix,imgpath in enumerate(images):
	im = scipy.misc.imread(imgpath.path,mode='YCbCr')/255
	h,w,c = im.shape
	if w%8 != 0 or h%8 != 0:
		newim = numpy.zeros((math.ceil(h/8)*8, math.ceil(w/8)*8,3))
		newim[0:h,0:w,:] = im
		newim[h:math.ceil(h/8)*8,0:w,:] = im[h-1:h,0:w,:]
		newim[0:h,w:math.ceil(w/8)*8,:] = im[0:h,w-1:w,:]
		newim[h:math.ceil(h/8)*8,w:math.ceil(w/8)*8,:] = im[h-1:h,w-1:w,:]
	else:
		newim = im

	for i in range(0,math.ceil(h/8)):
		for j in range(0,math.ceil(w/8)):
			block = newim[i*8:(i+1)*8,j*8:(j+1)*8,:]
			
			blockim = Image.fromarray(numpy.uint8(block*255),mode='YCbCr')
			inmem = io.BytesIO()
			q = random.randint(25,75)
			ss = random.randint(0,2)
			blockim.save(inmem,format='jpeg',quality=q,subsampling=ss)
			jpegblock = scipy.misc.imread(inmem,mode='YCbCr')/255



			# block = (block-0.5)/0.2
			# jpegblock = (jpegblock-0.5)/0.2

			# for k in range(0,3):
			# 	block[:,:,k] = numpy.divide((block[:,:,k]-0.5),0.2)
			# 	jpegblock[:,:,k] = numpy.divide((jpegblock[:,:,k]-0.5),0.2)

			# # too many 'plain colour' patches, filter some/all
			# if numpy.sqrt(numpy.mean((jpegblock-block)**2)) < 0.01 and random.randint(0,3) != 3:
			# 	cskip += 1
			# 	if not testing:
			# 		trainix -= 1
			# 	continue

			if count >= trainix:
				testing = True
				final = count
				count = 0
				print("\r{} training blocks{}".format(final,' '*50))

			if not testing:
				training_gt[count,:] = block.reshape((1,8*8*3))
				training_in[count,:] = jpegblock.reshape((1,8*8*3))
			else:
				testing_gt[count,:] = block.reshape((1,8*8*3))
				testing_in[count,:] = jpegblock.reshape((1,8*8*3))

			count += 1

	workdone = ix/len(images)
	print("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(workdone * 50), workdone*100), end="", flush=True)

if cskip != 0:
	training_gt = training_gt[0:final,:]
	training_in = training_in[0:final,:]
	testing_gt = testing_gt[0:count,:]
	testing_in = testing_in[0:count,:]

print("Means Y: {} Cb: {} Cr: {}".format(numpy.mean(training_in[:,0:64]),numpy.mean(training_in[:,64:128]),numpy.mean(training_in[:,128:])))
print("Vars Y: {} Cb: {} Cr: {}".format(numpy.var(training_in[:,0:64]),numpy.var(training_in[:,64:128]),numpy.var(training_in[:,128:])))

print("")
print("{} testing blocks".format(count))
print("{} blocks skipped".format(cskip))

print("Shuffling")
alltraining = numpy.concatenate([training_gt,training_in],axis=1)
numpy.random.shuffle(alltraining)
alltesting = numpy.concatenate([testing_gt,testing_in],axis=1)
numpy.random.shuffle(alltesting)

training_gt = alltraining[:,:8*8*3]
training_in = alltraining[:,8*8*3:]
testing_gt = alltesting[:,:8*8*3]
testing_in = alltesting[:,8*8*3:]

print("Saving data")

with open('data.pickle', 'wb') as file:
	numpy.savez_compressed(file,training_gt=training_gt,training_in=training_in,testing_gt=testing_gt,testing_in=testing_in)

