import csv
import numpy
import math
from random import randint
import sys
inputx = []
inputy = []
weight = []
gweight = []
gb = []
size = 0
feature = 0
testx = []
testy = []
#----------------------------------------------------------------------
def int_row(row):
	row1=[]
	for i in range(0,len(row)):
		row1.append(int(row[i]))
	return row1
#----------------------------------------------------------------------
def csv_reader(file_obj,arrlist):
    global data_size
    """
    Read a csv file
    """
    data_size = 0
    reader = csv.reader(file_obj)
    for row in reader:
        data_size += 1
        arrlist.append(int_row(row))    

#--------------------------------------------------------------------------
def read_data():
    global inputx,inputy,weight,size,feature
    csv_path = sys_arv[1]
    with open(csv_path) as f_obj:
        csv_reader(f_obj,inputx)
    inputx = numpy.array(inputx)

#--------------------------------------------------------------------------
def write_data():
    global inputx,inputy,size,feature
    file = open('output.txt','w')
    for i in range(0,size):
        file.write(str(0)+" ")
        for j in range(0,feature):
            file.write(str(j)+":"+str(inputx[i][j])+" ")
        file.write('\n')
    file.close()
    
#----------------------------------------------------------------------------    
read_data()
write_data()