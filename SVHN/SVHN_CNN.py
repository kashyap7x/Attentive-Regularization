from __future__ import print_function
import sys
sys.path.append('..')
import keras
from keras.optimizers import Adam
from denseBlocks import DenseNet
from visualization import *

import scipy.io as sio
import numpy as np
import h5py

batch_size = 64
num_classes = 10
epochs = 5

svhnTrain = sio.loadmat('train_32x32.mat')
svhnTest = sio.loadmat('test_32x32.mat')

xTrain = svhnTrain['X']
xTest  = svhnTest['X']
yTrain = svhnTrain['y']
yTest  = svhnTest['y']

# input image dimensions
img_rows, img_cols = 32, 32

x_train = xTrain
y_train = yTrain
x_test = xTest
y_test = yTest

#trainMean = np.mean(x_train, axis=0)

trainMeanFile = h5py.File('SVHN_train_mean.h5', 'r')
#trainMeanFile.create_dataset('train_mean', data=trainMean)
trainMean = np.array(trainMeanFile['train_mean'])
trainMeanFile.close()

x_train = np.transpose(x_train, (3, 0, 1, 2)) - trainMean
x_test = np.transpose(x_test, (3, 0, 1, 2)) - trainMean

print(x_train.shape)
print(x_test.shape)

# convert class vectors to binary class matrices
y_train -= 1
y_test -= 1
y_train = y_train.ravel()
y_test = y_test.ravel()
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

model = DenseNet(growth_rate=48, reduction=0.5, dropout_rate=0.0, include_target='false', l2=0.01, l2_buildup = 1)

# Optimizer
adam = Adam(lr= 0.001, beta_1= 0.9, beta_2= 0.999, epsilon= 1e-08, decay= 0.0)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=adam,
              metrics=['accuracy'])

# check the model

model.summary()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# visualization of the gaussian filters
# visualizeLayerOutput(model, 10, 4, 4)
# visualizeLayerOutput(model, 16, 4, 4)
# visualizeLayerOutput(model, 22, 4, 4)