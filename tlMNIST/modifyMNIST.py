import sys
from keras.datasets import mnist
from keras import backend as K
import numpy as np
from numpy.random import permutation
from numpy.random import uniform
np.random.seed(1337)
import h5py
import cv2

# input image dimensions
img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

tlMNIST_Train = np.ndarray((60000, 2 * img_rows, 2 * img_cols, 1), dtype='float32')

for i, train in enumerate(x_train):
    tlMNIST_Train[i, 0:img_rows, 0:img_cols] = train
    trainLabel = y_train[i]
    randPerm = uniform(0, 60000, 3)
    trImg = x_train[int(randPerm[0])]
    blImg = x_train[int(randPerm[1])]
    brImg = x_train[int(randPerm[2])]

    tlMNIST_Train[i, 0:img_rows, img_cols:] = trImg
    tlMNIST_Train[i, img_rows:, 0:img_cols] = blImg
    tlMNIST_Train[i, img_rows:, img_cols:] = brImg

    # if i<100:
    #     cv2.imshow('Sanity_check', np.array(tlMNIST_Train[i], dtype='uint8'));
    #     cv2.waitKey(30);
    #     cv2.destroyAllWindows();

trainFile = h5py.File('tlMNIST_train.h5', 'w')
trainDataset = trainFile.create_dataset('train_data', data=tlMNIST_Train)
trainDataset = trainFile.create_dataset('train_label', data=y_train)
trainFile.close()

tlMNIST_Test = np.ndarray((10000, 2 * img_rows, 2 * img_cols, 1), dtype='float32')

for i, test in enumerate(x_test):
    tlMNIST_Test[i, 0:img_rows, 0:img_cols] = test
    trainLabel = y_test[i]
    randPerm = uniform(0, 10000, 3)
    trImg = x_test[int(randPerm[0])]
    blImg = x_test[int(randPerm[1])]
    brImg = x_test[int(randPerm[2])]

    tlMNIST_Test[i, 0:img_rows, img_cols:] = trImg
    tlMNIST_Test[i, img_rows:, 0:img_cols] = blImg
    tlMNIST_Test[i, img_rows:, img_cols:] = brImg

testFile = h5py.File('tlMNIST_test.h5', 'w')
testDataset = testFile.create_dataset('test_data', data=tlMNIST_Test)
testDataset = testFile.create_dataset('test_label', data=y_test)
testFile.close()