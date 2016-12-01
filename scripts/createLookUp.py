#create a a word embedding matrix in the vocab word order

import gensim
import pickle
import numpy as np
import h5py

print "loading model..."
model = gensim.models.Word2Vec.load("/home/fbarbieri/compareEmojis/models/us/model_swm-300-6-3")
print "loading vocab..."
vocab = pickle.load(open( "/home/fbarbieri/deepbio/dataset/lfma/vocab.p", "rb" ))

vocab_size = len(vocab)
embedding_size = 300

lookup = np.zeros((vocab_size,embedding_size))
print "Starting ("+str(vocab_size)+")"
for i in range(vocab_size):
	if i%10000 == 0:
		print str(i)
	try:
		lookup[i] = model[vocab[i]]
	except:
		print "Didnt find: "+ vocab[i]

print "Saving..."
with h5py.File('/home/fbarbieri/deepbio/dataset/lfma/lookup.h5', 'w') as hf:
	hf.create_dataset('lookup', data=lookup)

print "Stai senza pensieri."