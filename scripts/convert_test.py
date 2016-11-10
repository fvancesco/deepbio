import h5py
import numpy as np
import sys

h5file = "/home/fbarbieri/deepbio/out/"+sys.argv[1]
myFile = h5py.File(h5file, 'r')
data = myFile['factors']
np.save(h5file.replace("h5","npy"), data)

print "Stai senza pensieri."