from collections import Counter
import operator
import numpy as np
import h5py
import pickle
import string

vocab_length = 100000
max_timeline_word = 800
n_users = 22113

timelines = open("/home/fbarbieri/deepbio/dataset/timelines_train.txt", 'r').read().strip().split("\n")
#timelines = open("/home/fbarbieri/mmtwitter/dataset/tmp.txt", 'r').read().strip().split("\n")
#create vocabulary
c = Counter()
for t in timelines:
	t = t.lower().translate(None, string.punctuation).strip()
	c.update(t.split(" "))

word_occ = sorted(c.items(), key=operator.itemgetter(1), reverse=True)
vocab = [w[0] for w in word_occ[:vocab_length]]
print "Vocabulary (100 most common):"
print vocab[0:100]
print "Vocabulary (100 most uncommon):"
print vocab[vocab_length-100:vocab_length]

w2i = {w:i for i,w in enumerate(vocab,1)} #start from 1 cause 0 is the padding value

#create tensor
tensor = np.zeros((n_users,max_timeline_word))

usersWithMoreW = 0
for u in range(n_users):
	#get most common words of current user
	tmp = timelines[u]
	t = tmp.lower().translate(None, string.punctuation).strip()

	'''
	c = Counter()
	c.update(t.split(" "))

	sw = sorted(c.items(), key=operator.itemgetter(1), reverse=True) #sorted words
	#print sw
	#fill tensor with indeces of words
	words = 0	
	for i in range(len(sw)):
		#print sw[i][0]
		index = w2i.get(sw[i][0])
		#print index
		if index is not None:
			tensor[u,words] = int(index)
			words+=1
			
		if words >= max_timeline_word : break
	'''
	
	words = 0	
	tokens = t.split(" ")
	#print len(tokens)
	for token in tokens:
		index = w2i.get(token)
#		if index is not None:
#			tensor[u,words] = int(index) #+1 #since unknown is the number 1 -- noooo!
#		else:
#			tensor[u,words] = 1 #unknown, do we want to add this? No sure
#		words+=1
		if index is not None:
			tensor[u,words] = int(index)
			words+=1
			
		if words >= max_timeline_word : 
			usersWithMoreW += 1
			break

	#print(tensor[u,:])

print "users with more words:"+str(usersWithMoreW)
#print(tensor)
#print(w2i)
#rint(timelines)

print "Saving timeline tensor..."
with h5py.File('/home/fbarbieri/deepbio/dataset/timelines_train.h5', 'w') as hf:
	hf.create_dataset('timelines', data=tensor)
	
print "Saving vocabulary mapping..."
pickle.dump(w2i, open("/home/fbarbieri/deepbio/dataset/w2i.p","wb"))

print "Saving vocabulary..."
pickle.dump(vocab, open("/home/fbarbieri/deepbio/dataset/vocab.p","wb"))	


	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
