import string
import numpy as np
data = ""
#data = "_entities"

print "------------Train--------------"
trainIndex = open("/home/fbarbieri/deepbio/dataset/lfma/items_index_train_LFMA.tsv", 'r').read().strip().split("\n")
trainOut = open("/home/fbarbieri/deepbio/dataset/lfma/timelines_train"+ data +".txt", 'w')
trainPath = "/home/fbarbieri/deepbio/dataset/lfma/bios"+ data +"/"
factors = np.load('/home/fbarbieri/deepbio/dataset/lfma/item_factors_als_200_LFMA.npy')
factors_new = np.zeros((199957,200))
f = 0
i = 0
for i in range(len(trainIndex)):
	try:
		timeline = open(trainPath+trainIndex[i]+".txt", 'r').read().strip().replace("\n"," ")
		tmp = timeline.lower().translate(None, string.punctuation).strip().split(" ")
		s = " ".join(set(tmp))
		trainOut.write(s+"\n")
		factors_new[f] = factors[i]
		f += 1
	except:
		print "didnt find " + trainIndex[i]+".txt"
	i += 1

np.save('/home/fbarbieri/deepbio/dataset/lfma/item_factors_als_200_LFMA_ok.npy', factors_new)
trainOut.close()


#------------------------------------------------------------------------
'''
print "------------Test--------------"
testIndex = open("/home/fbarbieri/deepbio/dataset/lfma/items_index_test_LFMA.tsv", 'r').read().strip().split("\n")
testOut = open("/home/fbarbieri/deepbio/dataset/lfma/timelines_test"+ data +".txt", 'w')
testPath = "/home/fbarbieri/deepbio/dataset/lfma/bios"+ data +"/"

for i in testIndex:
	try:
		timeline = open(testPath+i+".txt", 'r').read().strip().replace("\n"," ")
		tmp = timeline.lower().translate(None, string.punctuation).strip().split(" ")
		s = " ".join(set(tmp))
		testOut.write(s+"\n")
	except:
		print "didnt find " + testPath+i+".txt"

testOut.close()

print "Stai senza pensieri"
'''