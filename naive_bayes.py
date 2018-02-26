import numpy as np
import math
import random
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import sys

#initializing stemmer
tokenizer = RegexpTokenizer(r'\w+')
en_stop = set(stopwords.words('english'))
p_stemmer = PorterStemmer()

# function that takes an input file and performs stemming to generate the output file
def getStemmedDocument(input_raw_data):
	docs = input_raw_data
	for doc in docs:
		raw = doc.lower()
		raw = raw.replace("<br /><br />", " ")
		tokens = tokenizer.tokenize(raw)
		stopped_tokens = [token for token in tokens if token not in en_stop]
		stemmed_tokens = [p_stemmer.stem(token) for token in stopped_tokens]
		documentWords = ' '.join(stemmed_tokens)
	return documentWords


def read_data(file_name):
	file = open("data/" + file_name,"r")
	all_text = file.readlines()
	if len(all_text)==0:
		print "empty document"
	return all_text

training_data = read_data("train_sample.txt")
training_labels = read_data("train_sample_labels.txt")
test_data = read_data("test_sample_text.txt")
test_labels = read_data("test_sample_labels.txt")

def make_dict(training_data):
	count = 0
	vocab_dict = {}
	for document in training_data:
		words = document.split()
		for word in words:
			if not word in vocab_dict.keys():
				vocab_dict[word] = count
				count += 1
	return vocab_dict


label_dict = make_dict(training_labels)	
number_classes = len(label_dict)
vocab_dict = make_dict(training_data)


def make_matrix():
	global vocab_dict,training_data,training_labels,number_classes,label_dict
	naive_matrix = np.ones((len(vocab_dict),number_classes))
	num_words_in_class = np.full((1,number_classes),len(vocab_dict))
	label_freq = np.zeros(number_classes)
	for i in range(len(training_data)):
		label = label_dict[training_labels[i].split()[0]]
		label_freq[label] +=1
		words = training_data[i].split()
		for word in words:
			word_index = vocab_dict[word]
			naive_matrix[word_index][label] +=1
			num_words_in_class[0][label] +=1
	return (np.log(naive_matrix) - np.log(num_words_in_class)),label_freq


naive_matrix,label_freq = make_matrix()


def predict(test_document):
	global naive_matrix,label_freq,number_classes,vocab_dict
	max_sum = 0
	predicted_class = -1
	for class_ in range(number_classes):
		sums = math.log(label_freq[class_])
		words = test_document.split()
		for word in words:
			if word in vocab_dict.keys():
				word_index = vocab_dict[word]
				sums += naive_matrix[word_index][class_]
		if sums > max_sum or class_==0:
			max_sum = sums
			predicted_class = class_
	return predicted_class


def accuracy(test_documents,labels):
	correct = 0.0
	for i in range(len(test_documents)):
		predicted_class = predict(test_documents[i])
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

def confusion_matrix(test_documents,labels):
	correct = np.zeros((len(label_dict),len(label_dict)))
	for i in range(len(test_documents)):
		predicted_class = predict(test_documents[i])
		if labels[i].split()[0] in label_dict.keys():
			expected_class = label_dict[labels[i].split()[0]]
			correct[expected_class][predicted_class] += 1
	return correct


training_accuracy = accuracy(training_data,training_labels)
test_accuracy = confusion_matrix(test_data,test_labels)
# print test_accuracy
print test_accuracy
