import numpy as np
import math
import random
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import sys


#hyper_parameter
bigram_thresh = 3

#initializing stemmer
tokenizer = RegexpTokenizer(r'\w+')
en_stop = set(stopwords.words('english'))
p_stemmer = PorterStemmer()

# function that takes an input file and performs stemming to generate the output file
def getStemmedDocument(input_raw_data):
	docs = input_raw_data
	new_doc = []
	for doc in docs:
		raw = doc.lower()
		raw = raw.replace("<br /><br />", " ")
		tokens = tokenizer.tokenize(raw)
		stopped_tokens = [token for token in tokens if token not in en_stop]
		stemmed_tokens = [p_stemmer.stem(token) for token in stopped_tokens]
		documentWords = ' '.join(stemmed_tokens)
		new_doc.append(documentWords)
	return new_doc


def read_data(file_name):
	file = open("data/" + file_name,"r")
	all_text = file.readlines()
	if len(all_text)==0:
		print "empty document"
	return all_text

training_data = read_data("train_sample.txt")
training_data = getStemmedDocument(training_data)
training_labels = read_data("train_sample_labels.txt")
test_data = read_data("test_sample_text.txt")
test_data = getStemmedDocument(test_data)
test_labels = read_data("test_sample_labels.txt")

def make_dict(training_data,bigram):
	count = 0
	vocab_dict = {}
	bigram_dict = {}	
	for document in training_data:
		words = document.split()
		i = 0
		for word in words:
			if i > 0:
				bigram_word = words[i-1] + " " + word
				if not bigram_word in bigram_dict.keys():
					bigram_dict[bigram_word] = 1
				else:
					bigram_dict[bigram_word] +=1
			if not word in vocab_dict.keys():
				vocab_dict[word] = count
				count += 1
			i +=1
	if bigram:
		for key in bigram_dict.keys():
			if bigram_dict[key] >= bigram_thresh:
				vocab_dict[key] = count
				count += 1
	return vocab_dict
	

label_dict = make_dict(training_labels,0)	
number_classes = len(label_dict)
# vocab_dict = make_dict(training_data,0)
vocab_dict = make_dict(training_data,1)



def make_matrix(bigram):
	global vocab_dict,training_data,training_labels,number_classes,label_dict
	naive_matrix = np.ones((len(vocab_dict),number_classes))
	num_words_in_class = np.full((1,number_classes),len(vocab_dict))
	label_freq = np.zeros(number_classes)
	for i in range(len(training_data)):
		label = label_dict[training_labels[i].split()[0]]
		label_freq[label] +=1
		words = training_data[i].split()
		j = 0
		for word in words:
			if j>0 and bigram:
				bigram_word = words[j-1] + " " + word
				if bigram_word in vocab_dict.keys():
					word_index = vocab_dict[bigram_word]
					naive_matrix[word_index][label] +=1
			word_index = vocab_dict[word]
			naive_matrix[word_index][label] +=1
			num_words_in_class[0][label] +=1
			j +=1
	return (np.log(naive_matrix) - np.log(num_words_in_class)),label_freq


naive_matrix,label_freq = make_matrix(1)


def predict(test_document,bigram):
	global naive_matrix,label_freq,number_classes,vocab_dict
	max_sum = 0
	predicted_class = -1
	for class_ in range(number_classes):
		sums = math.log(label_freq[class_])
		words = test_document.split()
		j = 0
		for word in words:
			if j>0 and bigram:
				bigram_word = words[j-1] + " " + word
				if bigram_word in vocab_dict.keys():
					word_index = vocab_dict[bigram_word]
					sums += naive_matrix[word_index][class_]
			if word in vocab_dict.keys():
				word_index = vocab_dict[word]
				sums += naive_matrix[word_index][class_]
			j += 1
		if sums > max_sum or class_==0:
			max_sum = sums
			predicted_class = class_
	return predicted_class


def accuracy(test_documents,labels,bigram):
	correct = 0.0
	for i in range(len(test_documents)):
		predicted_class = predict(test_documents[i],bigram)
		if labels[i].split()[0] in label_dict.keys():
			expected_class = label_dict[labels[i].split()[0]]
			if predicted_class==expected_class:
				correct+=1
	return correct/len(test_documents)

def majority_accuracy(test_documents,labels):
	correct = 0.0
	for i in range(len(test_documents)):
		predicted_class = np.argmax(label_freq)
		if labels[i].split()[0] in label_dict.keys():
			expected_class = label_dict[labels[i].split()[0]]
			if predicted_class==expected_class:
				correct+=1
	return correct/len(test_documents)

def random_accuracy(test_documents,labels):
	correct = 0.0
	for i in range(len(test_documents)):
		predicted_class = random.randint(0,len(label_dict))
		if labels[i].split()[0] in label_dict.keys():
			expected_class = label_dict[labels[i].split()[0]]
			if predicted_class==expected_class:
				correct+=1
	return correct/len(test_documents)

def confusion_matrix(test_documents,labels,bigram):
	correct = np.zeros((len(label_dict),len(label_dict)))
	for i in range(len(test_documents)):
		predicted_class = predict(test_documents[i],bigram)
		if labels[i].split()[0] in label_dict.keys():
			expected_class = label_dict[labels[i].split()[0]]
			correct[expected_class][predicted_class] += 1
	return correct





# training_accuracy = accuracy(training_data,training_labels,1)
test_accuracy = confusion_matrix(test_data,test_labels,1)
# print test_accuracy
print test_accuracy
