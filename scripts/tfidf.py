import os, sys
import gensim
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import operator
import h5py
import pickle
import string, re

vocab_length = 200000 #150000 #100000
max_timeline_word = 800 #800 #600 
dataset = 'test' # test or train
data = "_tfidf" #"_entities"
i2w_file = "/home/fbarbieri/deepbio/dataset/i2w"+data+".p"
w2i_file = "/home/fbarbieri/deepbio/dataset/w2i"+data+".p"
w2tfidf_file = "/home/fbarbieri/deepbio/dataset/w2tfidf.p"

def tfidfVector(corpus):
	#vectorizer = TfidfVectorizer(min_df=1, stop_words=None, strip_accents=unicode)
	#vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, analyzer='word', max_features=vocab_length)
	vectorizer = TfidfVectorizer(sublinear_tf=True, analyzer='word', max_features=vocab_length)
	vectorizer.fit_transform(corpus)
	idfDic = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
	return idfDic

#---------------------------------------------------------------------

if 'train' in dataset:
	timelines = open("/home/fbarbieri/deepbio/dataset/timelines_train.txt", 'r').read().strip().split("\n")
	timelinesPath = "/home/fbarbieri/deepbio/dataset/bios/train/"
	h5_file = "/home/fbarbieri/deepbio/dataset/timelines_train"+data+".h5"

	print "loading corpus to calculate tfidf"
	corpus = []
	count = 0
	for f in os.listdir(timelinesPath):
		vector = open(timelinesPath+f, 'r').read().strip().replace("\n", " ")
		corpus.append(vector)
		if count%1000 == 0: print str(count)
		count += 1

	print "calculating tfidf"
	w2tfidf = tfidfVector(corpus)
	print 'w2tfidf: ' +  str(len(w2tfidf))
	count = 0
	i2w = []
	for key in w2tfidf.keys():
	  i2w.append(key)
	  count += 1
	print 'i2w:'+ str(count)
	w2i = {}
	for i in range(len(i2w)):
		w2i[i2w[i]] = i
	print 'i2w:'+ str(count)
	pickle.dump(w2tfidf, open(w2tfidf_file,"wb"))
	pickle.dump(i2w, open(i2w_file,"wb"))
	pickle.dump(w2i, open(w2i_file,"wb"))
elif 'test' in dataset:
	w2tfidf = pickle.load(open(w2tfidf_file, "rb"))
	i2w = pickle.load(open(i2w_file, "rb"))
	w2i = pickle.load(open(w2i_file, "rb"))
	timelines = open("/home/fbarbieri/deepbio/dataset/timelines_test.txt", 'r').read().strip().split("\n")
	h5_file = "/home/fbarbieri/deepbio/dataset/timelines_test"+data+".h5"

#create tensor for each user
n_users = len(timelines)
tensor = np.zeros((n_users,max_timeline_word))

usersWithMoreW = 0
for u in range(n_users):
	if u%1000 == 0: print str(u)
	#get most common words of current user
	t = timelines[u]
	#t = t.lower().translate(None, string.punctuation).strip()
	#t = ''.join([j if ord(j) < 128 else ' ' for j in t]) #rimuovi non ASCII del cazzo
	#t = re.sub(' +',' ',t) #rimuovi doppi spazi del cazzo   
	words = 0	
	tokens = t.split(" ")
	#print len(tokens)
	for token in tokens:
		#if token in i2w: #using try to go faster
		try:
			#index = i2w.index(token)
			index = w2i[token]
			tensor[u,words] = index
			words+=1	
		except:
			pass
		if words >= max_timeline_word : 
			usersWithMoreW += 1
			break

print "users with more words:"+str(usersWithMoreW)

print "Saving timeline tensor..."
with h5py.File(h5_file, 'w') as hf:
	hf.create_dataset('timelines', data=tensor)


	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	






