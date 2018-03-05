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

training_data = read_data("imdb_train_text.txt")
# training_data = getStemmedDocument(training_data)
training_labels = read_data("imdb_train_labels.txt")
test_data = read_data("imdb_test_text.txt")
# test_data = getStemmedDocument(test_data)
test_labels = read_data("imdb_test_labels.txt")

vocab_set = set()
for document in training_data:
	words = document.split()
	for word in words:
		vocab_set.add(word)
vocab_dict ={}
count = 0
for word in vocab_set:
	vocab_dict[word] = count
	count +=1





vocab_dict = {}
bigram_dict = {}
label_dict = {}

def make_dict(label,bigram):
	count = 0
	for document in training_data:
		words = document.split()
		i = 0
		for word in words:
			if i > 0 and bigram and not label:
				bigram_word = words[i-1] + " " + word
				if not bigram_word in bigram_dict.keys():
					bigram_dict[bigram_word] = 1
				else:
					bigram_dict[bigram_word] +=1
			if label:
				if not word in label_dict.keys():
					label_dict[word] = count
					count += 1
			else:
				if not word in vocab_dict.keys():
					vocab_dict[word] = count
					count += 1
			i +=1
	if bigram and not label:
		for key in bigram_dict.keys():
			if bigram_dict[key] >= bigram_thresh:
				vocab_dict[key] = count
				count += 1


make_dict(1,0)
make_dict(0,0)
number_classes = len(label_dict)


naive_matrix = np.ones((len(vocab_dict),number_classes))
num_words_in_class = np.full((1,number_classes),len(vocab_dict))
label_freq = np.zeros(number_classes)

def make_matrix(bigram):
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


make_matrix(0)


def predict(test_document,bigram):
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
