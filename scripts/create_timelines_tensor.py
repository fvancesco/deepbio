from collections import Counter
import operator
import numpy as np
import h5py
import pickle
import string

vocab_length = 100000
max_timeline_word = 800
dataset = 'test' # test or train

if 'train' in dataset:
	timelines = open("/home/fbarbieri/deepbio/dataset/timelines_train.txt", 'r').read().strip().split("\n")
	h5_file = "/home/fbarbieri/deepbio/dataset/timelines_train.h5"
	w2i_file = "/home/fbarbieri/deepbio/dataset/w2i_train.p"
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
	print "Saving vocabulary..."
	pickle.dump(vocab, open("/home/fbarbieri/deepbio/dataset/vocab.p","wb"))
elif 'test' in dataset:
	vocab = pickle.load(open("/home/fbarbieri/deepbio/dataset/vocab.p", "rb"))
	timelines = open("/home/fbarbieri/deepbio/dataset/timelines_test.txt", 'r').read().strip().split("\n")
	h5_file = "/home/fbarbieri/deepbio/dataset/timelines_test.h5"
	w2i_file = "/home/fbarbieri/deepbio/dataset/w2i_test.p"

w2i = {w:i for i,w in enumerate(vocab,1)} #start from 1 cause 0 is the padding value

#create tensor
n_users = len(timelines)
tensor = np.zeros((n_users,max_timeline_word))

usersWithMoreW = 0
for u in range(n_users):
	#get most common words of current user
	tmp = timelines[u]
	t = tmp.lower().translate(None, string.punctuation).strip()
	
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
with h5py.File(h5_file, 'w') as hf:
	hf.create_dataset('timelines', data=tensor)
	
print "Saving vocabulary mapping..."
pickle.dump(w2i, open(w2i_file,"wb"))


	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
