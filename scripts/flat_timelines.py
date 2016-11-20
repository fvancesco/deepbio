import string
#data = ""
data = "_entities"

trainIndex = open("/home/fbarbieri/deepbio/dataset/items_index_train_MSDA.tsv", 'r').read().strip().split("\n")
trainOut = open("/home/fbarbieri/deepbio/dataset/timelines_train"+ data +".txt", 'w')
trainPath = "/home/fbarbieri/deepbio/dataset/bios"+ data +"/train/"

for i in trainIndex:
	timeline = open(trainPath+i+".txt", 'r').read().strip().replace("\n"," ")
	tmp = timeline.lower().translate(None, string.punctuation).strip().split(" ")
	s = " ".join(set(tmp))
	trainOut.write(s+"\n")

trainOut.close()

#------------------------------------------------------------------------

testIndex = open("/home/fbarbieri/deepbio/dataset/items_index_test_MSDA.tsv", 'r').read().strip().split("\n")
testOut = open("/home/fbarbieri/deepbio/dataset/timelines_test"+ data +".txt", 'w')
testPath = "/home/fbarbieri/deepbio/dataset/bios"+ data +"/test/"

for i in testIndex:
	timeline = open(testPath+i+".txt", 'r').read().strip().replace("\n"," ")
	tmp = timeline.lower().translate(None, string.punctuation).strip().split(" ")
	s = " ".join(set(tmp))
	testOut.write(s+"\n")

testOut.close()

print "Stai senza pensieri"

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
