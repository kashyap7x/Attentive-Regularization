from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, Concatenate, MaxPooling2D, GlobalAveragePooling2D, Input, BatchNormalization, Activation, Flatten, Dropout
from keras import backend as K
from keras import regularizers
import sys
sys.path.append('..')
from layer import Target2D
from denseBlocks import DenseTarget2D, DenseNet
from visualization import *
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.callbacks import LearningRateScheduler, CSVLogger, ModelCheckpoint, TensorBoard
import h5py
import tensorflow as tf
np.random.seed(1337)

batch_size = 128
num_classes = 10
epochs = 15

def scheduler(epoch):
    if epoch==10 or epoch==15:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr*.1)
        print("lr changed to {}".format(lr*.1))
    return K.get_value(model.optimizer.lr)

# input image dimensions
img_rows, img_cols = 28, 28
scaling = 2
new_img_rows, new_img_cols = img_rows * scaling, img_cols * scaling

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

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#model = DenseNet(in_shape=input_shape, growth_rate=12, lay_per_block=2, reduction=0, dropout_rate=0, include_target='true', l2=0.0002, l2_buildup=1)

input = Input(shape=input_shape)
x = Target2D(256, kernel_size=(5, 5), padding='same', use_bias=False, attention_function='gaussian', sig1_regularizer=regularizers.l2(0.0001),sig2_regularizer=regularizers.l2(0.0001))(input)
#x = Conv2D(256, kernel_size=(5, 5), padding='same', use_bias=False)(input)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Target2D(256, kernel_size=(5, 5), padding='same', use_bias=False, attention_function='gaussian', sig1_regularizer=regularizers.l2(0.0002),sig2_regularizer=regularizers.l2(0.0002))(x)
#x = Conv2D(256, kernel_size=(5, 5), padding='same', use_bias=False)(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Target2D(128, kernel_size=(5, 5), padding='same', use_bias=False, attention_function='gaussian', sig1_regularizer=regularizers.l2(0.0004),sig2_regularizer=regularizers.l2(0.0004))(x)
#x = Conv2D(128, kernel_size=(5, 5), padding='same', use_bias=False)(x)
x = Activation('relu')(x)
x = Flatten()(x)

x = Dense(328, activation='relu')(x)
x = Dense(192, activation='relu')(x)
x = Dropout(0.5)(x)
out = Dense(num_classes, activation='softmax')(x)
model = Model(input, out)


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

keras.backend.get_session().run(tf.initialize_all_variables())

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=callbacks_list,
          validation_data=(x_test, y_test))

# visualization of the gaussian filters

visualizeLayerOutput(model, 1, 16, 16)
visualizeLayerOutput(model, 4, 16, 16)
visualizeLayerOutput(model, 7, 16, 8)
'''
visualizeLayerOutput(model, 8, 4, 3)
visualizeLayerOutput(model, 15, 4, 3)
visualizeLayerOutput(model, 26, 4, 3)
visualizeLayerOutput(model, 33, 4, 3)
visualizeLayerOutput(model, 44, 4, 3)
visualizeLayerOutput(model, 51, 4, 3)
'''