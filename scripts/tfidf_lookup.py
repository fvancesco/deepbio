#create a a word embedding matrix in the i2w word order

import gensim
import pickle
import numpy as np
import h5py


print "loading i2w..."
i2w = pickle.load(open( "/home/fbarbieri/deepbio/dataset/lfma/i2w_tfidf.p", "rb" ))
w2tfidf = pickle.load(open( "/home/fbarbieri/deepbio/dataset/lfma/w2tfidf.p", "rb" ))
#create tfidf table
i2w_size = len(i2w)
embedding_size = 1

didntfind = 0
lookup = np.random.rand(200000,embedding_size)
print "Starting ("+str(i2w_size)+")"
for i in range(i2w_size):
	if i%10000 == 0:
		print str(i)
	try:
		lookup[i] = w2tfidf[i2w[i]]
	except:
		print "Didnt find: "+ i2w[i]
		didntfind += 1

print 'didnt find ' + str(didntfind) + ' of ' + str(i2w_size)
print "Saving..."
with h5py.File('/home/fbarbieri/deepbio/dataset/lfma/tfidf.h5', 'w') as hf:
	hf.create_dataset('lookup', data=lookup)

#create lookup table
print "loading model..."
model = gensim.models.Word2Vec.load("/home/fbarbieri/compareEmojis/models/us/model_swm-300-6-3")
embedding_size = 300
didntfind = 0
lookup = np.random.rand(200000,embedding_size)
print "Starting ("+str(i2w_size)+")"
for i in range(i2w_size):
	if i%10000 == 0:
		print str(i)
	try:
		lookup[i] = model[i2w[i]]
	except:
		print "Didnt find: "+ i2w[i]
		didntfind += 1

print 'didnt find ' + str(didntfind) + ' of ' + str(i2w_size)

print "Saving..."
with h5py.File('/home/fbarbieri/deepbio/dataset/lfma/lookup_tfidf.h5', 'w') as hf:
	hf.create_dataset('lookup', data=lookup)

print "Stai senza pensieri."