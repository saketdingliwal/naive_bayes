def read_data(file_name):
	file = open("data/" + file_name,"r")
	all_text = file.readlines()
	if len(all_text)==0:
		print "empty document"
	return all_text




training_data = read_data("imdb_train_text.txt")
training_labels = read_data("imdb_train_labels.txt")



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
	for i in range(len(training_data)):
		label = label_dict[training_labels[i]]
		words = training_data[i].split()
		for word in words:
			word_index = vocab_dict[word]
			naive_matrix[word_index][label] +=1
			num_words_in_class[label] +=1
	return np.log(naive_matrix) - np.log(num_words_in_class)


naive_matrix = make_matrix()
