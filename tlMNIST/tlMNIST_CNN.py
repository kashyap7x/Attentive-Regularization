from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras import regularizers
import sys
sys.path.append('..')
from layer import Target2D
from visualization import *

from keras.layers import Concatenate, Input
from keras.models import Model
import h5py

batch_size = 128
num_classes = 10
epochs = 10

# input image dimensions
img_rows, img_cols = 28, 28
scaling = 2
new_img_rows, new_img_cols = img_rows * scaling, img_cols * scaling

# the data, shuffled and split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

trainFile = h5py.File('tlMNIST_train.h5', 'r')
testFile = h5py.File('tlMNIST_test.h5', 'r')

x_train = np.array(trainFile['train_data'])
y_train = np.array(trainFile['train_label'])

x_test = np.array(testFile['test_data'])
y_test = np.array(testFile['test_label'])

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, new_img_rows, new_img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, new_img_rows, new_img_cols)
    input_shape = (1, new_img_rows, new_img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], new_img_rows, new_img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], new_img_rows, new_img_cols, 1)
    input_shape = (new_img_rows, new_img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Target2D(attention_function='cauchy',
                   sig1_regularizer=regularizers.l2(0.01),
                   sig2_regularizer=regularizers.l2(0.01)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# check the model
model.summary()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# visualization of the gaussian filters
visualizeLayerOutput(model, 2, 8, 8)