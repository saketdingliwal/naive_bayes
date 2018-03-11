
# coding: utf-8

# In[77]:


import numpy as np
import math
import random
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
import nltk
import sys


#hyper_parameter
C = 1
bigram_thresh = 7
negation_thresh = 2
punct_list = [",",".","/","\""]
negate_list = ["not","no","never","didn't","nt"]


# In[78]:


nltk.download('stopwords')


# In[79]:


#initializing stemmer
tokenizer = RegexpTokenizer(r'\w+')
en_stop = set(stopwords.words('english'))
p_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


# In[80]:


def remove_punct(documents):
	new_documents = []
	for document in documents:
		for i in range(len(punct_list)):
			document = document.lower()
			document = document.replace(punct_list[i]," ")
		new_documents.append(document)
	return new_documents


# In[81]:


def getStemmedDocument(input_raw_data,adjective,negation_token):
    docs = input_raw_data
    new_doc = []
    for doc in docs:
        doc = doc.decode('utf-8')        
        raw = doc.lower()
        raw = raw.replace("<br /><br />", " ")
        raw.replace(" br "," ")
        tokens = tokenizer.tokenize(raw)
        if adjective>0 :
            pos = nltk.pos_tag(tokens)
            adj_list = [tag[0] for tag in pos if tag[1] == 'JJ']
        stopped_tokens = [token for token in tokens if token not in en_stop]
        stemmed_tokens = [p_stemmer.stem(token) for token in stopped_tokens]
        if adjective>0 :
            stemmed_adj_tokens = [p_stemmer.stem(token) for token in adj_list]
            for i in range(adjective):
                stemmed_tokens = stemmed_tokens + stemmed_adj_tokens
        if negation_token:
            stemmed_tokens = not_clear(stemmed_tokens)
        documentWords = ' '.join(stemmed_tokens)
        new_doc.append(documentWords)
    return new_doc


# In[82]:


def not_clear(tokens):
    i =0
    for token in tokens:
        if token in negate_list:
            if i+1 < len(tokens):
                tokens[i+1] = "not" + tokens[i+1]
            if i+2 < len(tokens):
                tokens[i+2] = "not" + tokens[i+2]
        i+=1
    return tokens


# In[83]:


def read_data(file_name):
	file = open("data/" + file_name)
	all_text = file.readlines()
	if len(all_text)==0:
		print "empty document"
	return all_text


# In[84]:


def make_dict(training_data,bigram):
	vocab_dict = {}
	bigram_dict = {}
	count = 0   
	for document in training_data:
		words = document.split()
		i = 0
		for word in words:
			if i > 0 and bigram:
				bigram_word = words[i-1] + " " + word
				if not bigram_word in bigram_dict:
					bigram_dict[bigram_word] = 1
				else:
					bigram_dict[bigram_word] +=1
			if not word in vocab_dict:
				vocab_dict[word] = count
				count += 1
			i +=1
	if bigram:
		for key in bigram_dict.keys():
			if bigram_dict[key] >= bigram_thresh:              
				vocab_dict[key] = count
				count += 1
	return vocab_dict


# In[85]:


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
				if bigram_word in vocab_dict:
					word_index = vocab_dict[bigram_word]
					sums += naive_matrix[word_index][class_]
			if word in vocab_dict:
				word_index = vocab_dict[word]
				sums += naive_matrix[word_index][class_]
			j += 1
		if sums > max_sum or class_==0:
			max_sum = sums
			predicted_class = class_
	return predicted_class


# In[86]:


def inv_label(label_val):
    for key in label_dict.keys():
        if label_dict[key] == label_val:
            return key
def indices(label_key):
    label_key = int(label_key)
    if label_key >= 7:
        label_key -= 2
    return label_key - 1


# In[87]:


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
		if labels[i].split()[0] in label_dict:
			expected_class = label_dict[labels[i].split()[0]]
			if predicted_class==expected_class:
				correct+=1
	return correct/len(test_documents)

def confusion_matrix(test_documents,labels,bigram):
	correct = np.zeros((len(label_dict),len(label_dict)))
	for i in range(len(test_documents)):
		predicted_class = predict(test_documents[i],bigram)
		predicted_class_key = inv_label(predicted_class)
		if labels[i].split()[0] in label_dict:
			expected_class_key = labels[i].split()[0]
			correct[indices(expected_class_key)][indices(predicted_class_key)] += 1
	return (correct)


# In[94]:


def idf_fill(training_data):
    idf_count = np.zeros(len(vocab_dict))
    for doc in training_data:
        words = doc.split()
        word_set = set()
        for word in words:
            word_set.add(word)
        for word in word_set:
            word_index = vocab_dict[word]
            idf_count[word_index] += 1
    idf_count =  np.log(len(training_data)) - np.log(idf_count)
    return idf_count


# In[89]:


def make_matrix(bigram,idf):
    naive_matrix = np.ones((len(vocab_dict),number_classes))
    naive_matrix = C * naive_matrix    
    num_words_in_class = np.full((1,number_classes),C*len(vocab_dict))
    label_freq = np.zeros(number_classes)
    for i in range(len(training_data)):
        label = label_dict[training_labels[i].split()[0]]
        label_freq[label] +=1
        words = training_data[i].split()
        j = 0
        for word in words:
            if j>0 and bigram:
                bigram_word = words[j-1] + " " + word
                if bigram_word in vocab_dict:
                    word_index = vocab_dict[bigram_word]
                    naive_matrix[word_index][label] +=1
            word_index = vocab_dict[word]
            naive_matrix[word_index][label] += (1 + idf*idf_count[word_index])
            num_words_in_class[0][label] += (1 + idf*idf_count[word_index])
            j +=1
    return (np.log(naive_matrix) - np.log(num_words_in_class)),label_freq


# In[96]:


feature = 3
training_data = read_data("imdb_train_text.txt")
if feature==2 or feature ==3 or feature ==5:
    training_data = getStemmedDocument(training_data,0,0)
elif feature==4:
    training_data = getStemmedDocument(training_data,0,1)
training_labels = read_data("imdb_train_labels.txt")
test_data = read_data("imdb_test_text.txt")
if feature==2 or feature ==3 or feature ==5:
    test_data = getStemmedDocument(test_data,0,0)
elif feature ==4:
    test_data = getStemmedDocument(test_data,0,1)
test_labels = read_data("imdb_test_labels.txt")
if feature == 3:
    bigram = 1
else:
    bigram = 0
label_dict = make_dict(training_labels,0)
vocab_dict = make_dict(training_data,bigram)
number_classes = len(label_dict)
idf_count = np.zeros(len(vocab_dict))
if feature ==5:
    idf_count  = idf_fill(training_data)
    naive_matrix,label_freq = make_matrix(bigram,1)
else:
    naive_matrix,label_freq = make_matrix(bigram,0)
training_accuracy = accuracy(training_data,training_labels,bigram)
print training_accuracy*100
test_accuracy = accuracy(test_data,test_labels,bigram)
print test_accuracy*100
confuse = confusion_matrix(test_data,test_labels,bigram)
for i in range(len(confuse)):
    for j in range(len(confuse[0])):
        print (int)(confuse[i][j]),
    print
print "================================================================================"


# In[92]:


# print "normal"
# feature_selection(1)
# print "stemmed"
# feature_selection(2)
# print "bigram"
# feature_selection(3)
# print "negation"
# feature_selection(4)
# print "idf"
# feature_selection(5)

