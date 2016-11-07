mport os
import sys

pathin = sys.argv[1]
pathout = sys.argv[2]

for f in os.listdir(pathin):
        if "_" in f:
                print "processing: "+f
                lines = open(pathin+f, 'r').read().strip().split("\n")

                f = f.replace(":","_")
                f = f.replace("-","_")
                f = f.replace(".txt","")

                #create output files
                out_trainloss = open(pathout+f+'__1_trainLoss.csv', 'w')
                out_valloss = open(pathout+f+'__2_valLoss.csv', 'w')
                #out_minibatchtrainloss = open(pathout+f+'__3_miniBatchTrainLoss.csv', 'w')
                out_ranktext = open(pathout+f+'__4_rankText'+'.csv', 'w')
                out_rankvis = open(pathout+f+'__5_rankVis'+'.csv', 'w')
                out_bestrankboth = open(pathout+f+'__6_bestRankBoth'+'.csv', 'w')
                out_cosine = open(pathout+f+'__7_cosine'+'.csv', 'w')

                #write first line
                out_trainloss.write("x,y\n")
                out_valloss.write("x,y\n")
                #out_minibatchtrainloss.write("x,y\n")
                out_ranktext.write("x,y\n")
                out_rankvis.write("x,y\n")
                out_bestrankboth.write("x,y\n")
                out_cosine.write("x,y\n")

                #exctract content
                start = False
                for l in lines:
                        if "total number of" in l: start=True
                        if "e:" in l and start:
                                epoch = l.split("e:")[1].split(" ")[0]

                                out_trainloss.write(epoch + "," + l.split("loss:")[1].split("/")[0].replace(" ","")+"\n")
                                out_valloss.write(epoch + "," + l.split("loss:")[1].split("/")[1].replace("text","")+"\n")
                                out_ranktext.write(epoch + "," + l.split("rank:")[1].split("(")[0].replace(" ","")+"\n")
                                out_rankvis.write(epoch + "," + l.split("rank:")[1].split("(")[1].split("/")[1]+"\n")
                                out_bestrankboth.write(epoch + "," + l.split("BC:")[1].split(" ")[0]+"\n")
                                out_cosine.write(epoch + "," + l.split("sim:")[1].split(" ")[1]+"\n")

                out_trainloss.close()
                out_valloss.close()
                #out_minibatchtrainloss.close()
                out_ranktext.close()
                out_rankvis.close()
                out_bestrankboth.close()
                out_cosine.close()
