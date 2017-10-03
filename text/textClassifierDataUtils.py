import os
import numpy as np
import sys
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras import backend as K
import matplotlib.pyplot as plt

def loadTexts(dirPath):
	'''
  Load text data from a directory and process into data and label vectors.

  Inputs:
  - dirPath: String giving path to the directory to load.

  Returns: A tuple of
  - data: List of strings from dataset
  - labels: Categorical class matrix form of label vector
  '''
	# Create lists for samples, label IDs and dictionary mapping label
	# names to numeric IDs
	data = []
	labels = []
	labelIndex = {}

	# Load the data from the directory
	for name in sorted(os.listdir(dirPath)):
		path = os.path.join(dirPath, name)
		if os.path.isdir(path):
			label_ID = len(labelIndex)
			labelIndex[name] = label_ID
			for fileName in sorted(os.listdir(path)):
				if fileName.isdigit():
					filePath = os.path.join(path, fileName)
					if sys.version_info < (3,):
						f = open(filePath)
					else:
						f = open(filePath, encoding='latin-1')
					data.append(f.read())
					f.close()
					labels.append(label_ID)

	# Convert class vector to class matrix
	labels = to_categorical(np.asarray(labels))
	return data, labels

def load20NewsData(trainDir, testDir, numWords, seqLen, dType, valSplit):
	'''
  Load and preprocess the 20 Newsgroups text classification dataset.
  
  Inputs:
  - trainDir: String giving path to the training set directory.
  - testDir: String giving path to the test set directory.
  - numWords: Number of highest frequency words to index.
  - seqLen: Word length of the data vectors to pad or truncate to.
  - dType: Data type for loading data vectors.
  - valSplit: Ratio (0-1) of training data to use for validation.

  Returns: A tuple of
  - X_train: Array of training sequences
  - Y_train: Array of training labels (categorical class matrix form)
  - X_val: Array of validation sequences (based on valSplit)
  - Y_val: Array of validation labels (categorical class matrix form)
  - X_test: Array of test sequences
  - Y_test: Array of test labels (categorical class matrix form)
  - wordIndex: Dictionary used for word to sequence mapping
  '''
	# Call loadTexts for train data
	trainData, trainLabels = loadTexts(trainDir)

	# Convert list of strings to sequences of integers padded to seqLen
	tokenizer = Tokenizer(nb_words=numWords)
	tokenizer.fit_on_texts(trainData)
	trainSequences = tokenizer.texts_to_sequences(trainData)
	trainDataPadded = pad_sequences(trainSequences, maxlen=seqLen)

	# Splitting into a training set and a validation set
	indices = np.arange(trainDataPadded.shape[0])
	np.random.shuffle(indices)
	trainDataPadded = trainDataPadded[indices]
	trainLabels = trainLabels[indices]
	numValSamples = int(valSplit * trainDataPadded.shape[0])

	X_train = (trainDataPadded[:-numValSamples]).astype(dType)
	Y_train = trainLabels[:-numValSamples]
	X_val = (trainDataPadded[-numValSamples:]).astype(dType)
	Y_val = trainLabels[-numValSamples:]

	# Call loadTexts for test data
	testData, Y_test = loadTexts(testDir)

	# Use dictionary of mappings from training to convert to sequences
	wordIndex = tokenizer.word_index
	# Loop over strings in list
	for ind, string in enumerate(testData):
		testSequences = text_to_word_sequence(string)
		# Loop over words in string
		for idx, word in enumerate(testSequences):
			testSequences[idx] = wordIndex.get(testSequences[idx], numWords + 1)
		testData[ind] = [value for value in testSequences if value <= numWords]
	X_test = pad_sequences(testData, maxlen=seqLen).astype(dType)

	return X_train, Y_train, X_val, Y_val, X_test, Y_test, wordIndex

def loadEmbeddings(dirPath, wordIndex, numWords):
	'''
	Load the GloVe 6B 100-dimensional embeddings from its directory.

	Inputs:
	- dirPath: String giving path to the directory to load.
	- wordIndex: Dictionary to use for word to sequence mapping
	- numWords: Number of highest frequency words to index.

	Returns:
	- embeddingMatrix: Array used to initialize the keras embedding layer.
	- numWordsActual: First dimension length for the matrix
	'''
	# Indexing of the embedding set from its words to embedding vectors
	index = {}

	f = open(os.path.join(dirPath, 'glove.6B.100d.txt'), encoding="utf8")
	for line in f:
		values = line.split()
		word = values[0]
		coefficients = np.asarray(values[1:], dtype='float32')
		index[word] = coefficients
	f.close()

	# Initialize to the right size
	numWordsActual = min(numWords, len(wordIndex))
	embeddingMatrix = np.zeros((numWordsActual + 1, 100))
	for word, i in wordIndex.items():
		if i > numWords:
			continue
		embeddingVector = index.get(word)
		if embeddingVector is not None:
			# Use zero initialization for words not found
			embeddingMatrix[i] = embeddingVector

	return embeddingMatrix, numWordsActual