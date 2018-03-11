
# coding: utf-8

# In[1]:
import sys
import pickle
import os
import math
import numpy as np
import random
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
import nltk

tokenizer = RegexpTokenizer(r'\w+')
en_stop = set(stopwords.words('english'))
p_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def predict(test_document,bigram):
	global naive_matrix,label_freq,vocab_dict,label_dict
	number_classes = len(label_dict)
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


# In[4]:


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


# In[3]:


def load_model(filename):
    with open(filename, 'rb') as f:
        vocab_dict = pickle.load(f)
        label_dict = pickle.load(f)
        naive_matrix = pickle.load(f)
        label_freq = pickle.load(f)
    return vocab_dict,label_dict,naive_matrix,label_freq


# In[ ]:


def read_data(file_name):
	file = open(file_name)
	all_text = file.readlines()
	if len(all_text)==0:
		print "empty document"
	return all_text


# In[5]:


def inv_label(label_val):
    for key in label_dict.keys():
        if label_dict[key] == label_val:
            return key


# In[ ]:


feature = int(sys.argv[1])
input_file = sys.argv[2]
output_file = sys.argv[3]
model_name = "./pickles/naive_model" + str(feature) + ".pkl"
vocab_dict,label_dict,naive_matrix,label_freq = load_model(model_name)
test_data = read_data(input_file)
if feature == 3:
    bigram = 1
else:
    bigram = 0

if not feature == 1:
    test_data = getStemmedDocument(test_data,0,0)

file = open(output_file,'wb')
for doc in test_data:
    predicted_label = inv_label(predict(doc,bigram))
    file.write(predicted_label+'\n')
