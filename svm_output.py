import os
import math
import sys
import numpy as np
from numpy import genfromtxt
import pickle


input_file = sys.argv[1]
output_file = sys.argv[2]
model_name = "./pickles/svm1" + ".pkl"

def predict_label(test_X):
	test_X = np.append(test_X,[1])
	labels_array = [0,1,2,3,4,5,6,7,8,9]
	labels = len(labels_array)
	wins = np.zeros((labels))
	for i in range(labels):
		for j in range(i+1,labels):
			dist = np.matmul(test_X,classifier[i][j])
			if dist > 0:
				wins[i] += 1
			elif dist < 0:
				wins[j] += 1
	label = -1
	maxx = -1
	for i in range(labels):
		if wins[i] >= maxx:
			maxx = wins[i]
			label = i
	return label

def read_data(input_file):
	X = genfromtxt(input_file,delimiter = ',') # list of training example vectors
	X = np.array(X,dtype="float64")
	X = X/255.0
# 	x0 = np.ones((len(X),1))
# 	X = np.hstack((x0,X))
	return X


with open(model_name, 'rb') as f:
    classifier = pickle.load(f)

test_data = read_data(input_file)
file = open(output_file,'wb')
for i in range(len(test_data)):
    predicted_label = (predict_label(test_data[i]))
    file.write(str(predicted_label)+'\n')
