from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.layers import Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Concatenate, Input, BatchNormalization, Activation
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam
from layer import Scale, Target2D, DenseTarget2D
from visualization import *

import scipy.io as sio
import numpy as np
import cv2
import h5py

batch_size = 128
num_classes = 10
epochs = 30

svhnTrain = sio.loadmat('/home/suriya/Documents/second_partition/Data/SVHN/train_32x32.mat');
svhnTest = sio.loadmat('/home/suriya/Documents/second_partition/Data/SVHN/test_32x32.mat');

print(len(svhnTrain))
print(len(svhnTest))

for key, value in svhnTrain.items():
    print(key)
for key, value in svhnTest.items():
    print(key)

xTrain = svhnTrain['X'];
xTest  = svhnTest['X'];
yTrain = svhnTrain['y'];
yTest  = svhnTest['y'];

print(xTrain.shape);
print(xTest.shape);
print(yTrain.shape);
print(yTest.shape);


# input image dimensions
img_rows, img_cols = 32, 32

x_train = xTrain;
y_train = yTrain;
x_test = xTest;
y_test = yTest;

#trainMean = np.mean(x_train, axis=0);
#testMean = np.mean(y_train, axis=0);

trainMeanFile = h5py.File('SVHN_train_mean.h5', 'r');
#trainMeanFile.create_dataset('train_mean', data=trainMean);
trainMean = np.array(trainMeanFile['train_mean']);
trainMeanFile.close();

testMeanFile = h5py.File('SVHN_test_mean.h5', 'r');
#testMeanFile.create_dataset('test_mean', data=testMean);
testMean = np.array(testMeanFile['test_mean']);
testMeanFile.close();

x_train = np.transpose(x_train, (3, 0, 1, 2)) - trainMean;
x_test = np.transpose(x_test, (3, 0, 1, 2)) - testMean;

print();
print(x_train.shape);
print(x_test.shape);
print();


if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

# convert class vectors to binary class matrices
print(y_train)
y_train -= 1;
y_test -= 1;
y_train = y_train.ravel();
y_test = y_test.ravel();
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

growth_rate=16
l2 = 0.001
l2_buildup = 5

input = Input(shape=input_shape)
x = Conv2D(32, kernel_size=(3, 3), padding='same', use_bias=False)(input)
x = BatchNormalization()(x)
x = Scale()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = DenseTarget2D(x, growth_rate=growth_rate, include_target = 'true', l2=l2)
l2 *= l2_buildup
x = DenseTarget2D(x, growth_rate=growth_rate, include_target = 'true', l2=l2)
l2 *= l2_buildup
x = DenseTarget2D(x, growth_rate=growth_rate, include_target = 'true', l2=l2)

x = BatchNormalization()(x)
x = Scale()(x)
x = Activation('relu')(x)
x = GlobalAveragePooling2D()(x)
out = Dense(num_classes, activation='softmax')(x)
model = Model(input, out)

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
visualizeLayerOutput(model, 10, 4, 4)
visualizeLayerOutput(model, 16, 4, 4)
visualizeLayerOutput(model, 22, 4, 4)