#hyperParameters
C = 1.0
T = 100000
k = 100



def read_data(input_file):
	X = genfromtxt(input_file,delimiter = ',') # list of training example vectors
	Y = X[:,len(X[0])-1]
	Y = Y[np.newaxis]
    Y = np.transpose(Y)
	X = np.delete(X,len(X[0]-1),1)
	x0 = np.ones((len(X),1))
	X = np.hstack((x0,X))
	return X,Y


X,Y = read_data('data/train.csv')




def pegasos(X_part,Y_part):
	m = len(X_part)
	k = min(m,k)
	w = np.zeros((len(X_part[0]),1))
	for t in range(T):
		k_array = np.random.choice(m,k)
		step_neta = C * (1/(t+1))
		update_amount = np.zeros((len(X_part[0]),1))
		for i in range(k):
			X_i = X_part[k_array[i]]
			X_w = np.matmul(X_i,w)
			y_X_w = Y_part[k_array[i]] * X_w[0][0]
			if y_X_w < 1:
				update_amount += Y_part[k_array[i]] * np.transpose(X_i)
		w = (1 - step_neta*(1/C)) * w + (step_neta/k) * update_amount 
	return w	

def make_dict(Y):
	pass
	# make label dictionary


def learn_parameters():
	labels_array = np.unique(Y)
	labels = len(labels_array)
	classifier = np.zeros((labels,labels,len(X[0]),1)))
	for i in range(labels):
		for j in range(i+1,labels):
			X_part = []
			Y_part = []
			for k in range(m):
				if Y[k]==i or Y[k]==j:
					X_part.append(X[k])
					if Y[k]==i:
						Y_part.append(1)
					else:
						Y_part.append(-1)
			w = pegasos(X_part,Y_part)
			classifier[i][j] = w
			classifier[j][i] = w
	return classifier

def predict_label(test_X):
	test_X = np.append([1],test_X)
	labels_array = np.unique(Y)
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


def accuracy(test_data,test_labels):
	count = 0
	for i in range(len(test_data)):
		predicted_label = predict_label(test_data[i])
		expected_label = test_labels[i]
		if predicted_label == expected_label:
			count += 1
	return (count/len(test_data))




