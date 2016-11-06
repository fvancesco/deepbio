
trainIndex = open("/home/fbarbieri/deepbio/dataset/items_index_train_MSDA.tsv", 'r').read().strip().split("\n")
trainOut = open("/home/fbarbieri/deepbio/dataset/timelines_train.txt", 'w')
trainPath = "/home/fbarbieri/deepbio/dataset/bios/train/"

for i in trainIndex:
	timeline = open(trainPath+i+".txt", 'r').read().strip().replace("\n"," ").split(" ")
	s = " ".join(set(timeline))
	trainOut.write(s+"\n")

print "Stai senza pensieri"
trainOut.close()



	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
