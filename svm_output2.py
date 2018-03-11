import csv
import numpy
import math
import sys
import numpy as np
from numpy import genfromtxt


def read_data(input_file):
	X = genfromtxt(input_file,delimiter = ',') # list of training example vectors
	X = np.array(X)
	return X,len(X),len(X[0])

def write_data(inp, size, feature):
    file = open('changed_format.txt','w')
    for i in range(0,size):
        file.write(str(0)+" ")
        for j in range(0,feature):
            file.write(str(j)+":"+str(inp[i][j])+" ")
        file.write('\n')
    file.close()


inputx, size, feature = read_data(sys.argv[1])
write_data(inputx, size, feature)
