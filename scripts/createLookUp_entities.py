#filter nasari in order to get only vocab words, then build a model in gensim, and create the lookup table. 
#saludos,
#francesco

import pickle
import numpy as np
import h5py

data = "_entities"
vocab = pickle.load(open( "/home/fbarbieri/deepbio/dataset/vocab"+data+".p", "rb" ))
vocab_size = len(vocab)
embedding_size = 300
lookup = np.zeros((vocab_size,embedding_size))

#------------------------------------------
#FILTER NASARI (needed to split, not enough memory to load at once)

print "loading nasari 1 txt..."
nasari = open("/home/fbarbieri/deepbio/dataset/nasari/NASARI_embed_english_1.txt", 'r').read().strip().split("\n")
n = 0
total = 2210150
for l in nasari[1:]:
	tokens = l.split(" ")
	word = tokens[0].split("__")[0].replace(":","")
	try:
		idx = vocab.index(word) #if the word does not exists then exept!
		vectorStr = ",".join(tokens[1:])
		vector = np.array(vectorStr.split(","), dtype=np.float32)
		lookup[idx]=vector
		n += 1
	except:
		pass
	total -= 1
	if total%10000 == 0:
		print str(total) + " left..."

del nasari

print "loading nasari 2 txt..."
nasari = open("/home/fbarbieri/deepbio/dataset/nasari/NASARI_embed_english_2.txt", 'r').read().strip().split("\n")
nasari_new = ""
total = 2210150
for l in nasari:
	tokens = l.split(" ")
	word = tokens[0].split("__")[0].replace(":","")
	try:
		idx = vocab.index(word) #if the word does not exists then exept!
		vectorStr = ",".join(tokens[1:])
		vector = np.array(vectorStr.split(","), dtype=np.float32)
		lookup[idx]=vector
		n += 1
	except:
		pass
	total -= 1
	if total%10000 == 0:
		print str(total) + " left..."

print "Saving..."
with h5py.File('/home/fbarbieri/deepbio/dataset/lookup'+data+'.h5', 'w') as hf:
	hf.create_dataset('lookup', data=lookup)

print "stai senza pensieri"