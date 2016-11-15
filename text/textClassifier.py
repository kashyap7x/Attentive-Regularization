'''
CNN for 20 Newsgroups classification, implemented on Keras with Theano backend.
Dataset Description - http://qwone.com/~jason/20Newsgroups/
Data - http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz
GloVe Embeddings Description - http://nlp.stanford.edu/projects/glove/
Embeddings - http://nlp.stanford.edu/data/glove.6B.zip
'''

from __future__ import print_function
from textClassifierDataUtils import load20NewsData, loadEmbeddings
import numpy as np
import matplotlib.pyplot as plt
import theano

from keras.models import Model
from keras.layers import Dense, Dropout, Input, Flatten, merge
from keras.layers import Conv1D, MaxPooling1D, Embedding, AttentiveRegularization1D
from keras.optimizers import Adam

baseDir = '/home/kashyap/keras-master'
gloveDir = baseDir + '/glove.6B/'
trainDir = baseDir + '/20news-bydate-train/'
testDir = baseDir + '/20news-bydate-test/'

valSplit = 0.1
seqLen = 500
numWords = 20000
dType = 'float32'

batchSize = 32
numEpoch = 20

# Load the data
X_train, Y_train, X_val, Y_val, X_test, Y_test, wordIndex = load20NewsData (trainDir, testDir, numWords, seqLen, dType, valSplit)

# Check data shapes
print ('Train data shape: ', X_train.shape)
print ('Train labels shape: ', Y_train.shape)
print ('Validation data shape: ', X_val.shape)
print ('Validation labels shape: ', Y_val.shape)
print ('Test data shape: ', X_test.shape)
print ('Test labels shape: ', Y_test.shape)

# Build the embedding matrix
embeddingMatrix, numWordsActual = loadEmbeddings (gloveDir, wordIndex, numWords)
embeddingLayer = Embedding(numWordsActual + 1,
                            100,
                            weights=[embeddingMatrix],
                            input_length=seqLen,
                            trainable=True)

# Model
inputSequences = Input(shape=(seqLen,), dtype='int32')
embeddedVectors = embeddingLayer(inputSequences)
x = Conv1D(256, 3, activation='relu', border_mode='same')(embeddedVectors)
x = AttentiveRegularization1D()(x)
x = MaxPooling1D(seqLen)(x)
x = Dropout(0.5)(x)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(20, activation='softmax')(x)
model = Model(inputSequences, predictions)

# Optimizer
adam = Adam(lr= 0.001, beta_1= 0.9, beta_2= 0.999, epsilon= 1e-08, decay= 0.0)

# Compile and train
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

'''
# Display the attention layer initialization
function = model.layers[3].function
plt.imshow(function.eval(), cmap='gray')
plt.show()
'''

model.fit(X_train, Y_train,
	batch_size=batchSize,
	nb_epoch=numEpoch,
	validation_data=(X_val, Y_val),
	shuffle=True)

# Evaluate test accuraccy
print (model.evaluate(X_test, Y_test, batch_size=batchSize))

'''
# Display the attention layer final values
function = model.layers[3].function
plt.imshow(function.eval(), cmap='gray')
plt.show()
'''
