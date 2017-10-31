from __future__ import print_function
import sys
sys.path.append('..')
import keras
from keras.datasets import mnist
from keras.layers import Dense, Conv2D, Concatenate, MaxPooling2D, GlobalAveragePooling2D, Input, BatchNormalization, Activation
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam, SGD
from denseBlocks import DenseTarget2D, DenseNet
from visualization import *
from keras.callbacks import LearningRateScheduler, CSVLogger, ModelCheckpoint, TensorBoard
np.random.seed(1337)

batch_size = 128
num_classes = 10
epochs = 20

def scheduler(epoch):
    if epoch==10 or epoch==15:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr*.1)
        print("lr changed to {}".format(lr*.1))
    return K.get_value(model.optimizer.lr)

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
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
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = DenseNet(in_shape=input_shape, growth_rate=12, lay_per_block=2, reduction=0.5, dropout_rate=0, include_target='true', l2=0.0001)

# Optimizer
#adam = Adam(lr= 0.001, beta_1= 0.9, beta_2= 0.999, epsilon= 1e-08, decay= 0.0)
sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=sgd,
              metrics=['accuracy'])

# check the model

model.summary()

lr_decay = LearningRateScheduler(scheduler)
csv_logger = CSVLogger('MNIST_train.log')
checkpoint = ModelCheckpoint(filepath='MNIST_weights.hdf5', monitor='val_acc', verbose=0, save_best_only=True, mode='max')
tensorBoard = TensorBoard(log_dir='./logs')
callbacks_list = [lr_decay, csv_logger, checkpoint, tensorBoard]

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=callbacks_list,
          validation_data=(x_test, y_test))

# visualization of the gaussian filters
visualizeLayerOutput(model, 8, 3, 4)
visualizeLayerOutput(model, 15, 3, 4)
visualizeLayerOutput(model, 26, 3, 4)
visualizeLayerOutput(model, 33, 3, 4)
visualizeLayerOutput(model, 44, 3, 4)
visualizeLayerOutput(model, 51, 3, 4)