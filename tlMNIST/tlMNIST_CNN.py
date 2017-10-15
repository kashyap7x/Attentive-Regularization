from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.layers import Dense, Conv2D, Concatenate, MaxPooling2D, GlobalAveragePooling2D, Input, BatchNormalization, Activation
from keras import backend as K
from keras import regularizers
import sys
sys.path.append('..')
from denseBlocks import DenseTarget2D
from visualization import *
from keras.models import Model
from keras.optimizers import Adam
import h5py

batch_size = 512
num_classes = 10
epochs = 100

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

growth_rate=16
l2 = 0.01
l2_buildup = 1

input = Input(shape=input_shape)
x = Conv2D(32, kernel_size=(3, 3), padding='same', use_bias=False)(input)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

y = DenseTarget2D(x, growth_rate=growth_rate, include_target = 'true', l2=l2)
x = Concatenate(axis=-1)([x, y])
l2 *= l2_buildup

y = DenseTarget2D(x, growth_rate=growth_rate, include_target = 'true', l2=l2)
x = Concatenate(axis=-1)([x, y])
l2 *= l2_buildup

y = DenseTarget2D(x, growth_rate=growth_rate, include_target = 'true', l2=l2)
x = Concatenate(axis=-1)([x, y])

x = BatchNormalization()(x)
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
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

'''
# visualization of the gaussian filters
visualizeLayerOutput(model, 10, 4, 4)
visualizeLayerOutput(model, 16, 4, 4)
visualizeLayerOutput(model, 22, 4, 4)
'''