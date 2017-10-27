from __future__ import print_function
import sys
sys.path.append('..')
import keras
from keras.optimizers import Adam, SGD
from keras.layers import Dense, Conv2D, Concatenate, MaxPooling2D, GlobalAveragePooling2D, Input, BatchNormalization, Activation
from keras.models import Model
from denseBlocks import DenseTarget2D
from denseBlocks import DenseNet, wideResNet
from SVHNDataUtils import getSVHNData
from visualization import *
from keras.callbacks import LearningRateScheduler, CSVLogger, ModelCheckpoint
np.random.seed(1337)

batch_size = 64
epochs = 40

def scheduler(epoch):
    if epoch==20 or epoch==30:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr*.1)
        print("lr changed to {}".format(lr*.1))
    return K.get_value(model.optimizer.lr)

x_train, y_train, x_val, y_val, x_test, y_test = getSVHNData()

#model = wideResNet(k=8, dropout=0.4, include_target='false')
model = DenseNet(growth_rate=48, reduction=0.5, dropout_rate=0.2, include_target='false')

'''
growth_rate=48
l2 = 0.001
l2_buildup = 1

input = Input(shape=(32,32,3))
x = Conv2D(32, kernel_size=(3, 3), padding='same', use_bias=False)(input)
x = BatchNormalization()(x)
x = Activation('relu')(x)

y = DenseTarget2D(x, growth_rate=growth_rate, include_target = 'true', l2=l2)
x = Concatenate(axis=-1)([x, y])
l2 *= l2_buildup

y = DenseTarget2D(x, growth_rate=growth_rate, include_target = 'true', l2=l2)
x = Concatenate(axis=-1)([x, y])
l2 *= l2_buildup

x = MaxPooling2D(pool_size=(2, 2))(x)

y = DenseTarget2D(x, growth_rate=growth_rate, include_target = 'true', l2=l2)
x = Concatenate(axis=-1)([x, y])
l2 *= l2_buildup

y = DenseTarget2D(x, growth_rate=growth_rate, include_target = 'true', l2=l2)
x = Concatenate(axis=-1)([x, y])
l2 *= l2_buildup

x = MaxPooling2D(pool_size=(2, 2))(x)

y = DenseTarget2D(x, growth_rate=growth_rate, include_target = 'true', l2=l2)
x = Concatenate(axis=-1)([x, y])
l2 *= l2_buildup

y = DenseTarget2D(x, growth_rate=growth_rate, include_target = 'true', l2=l2)
x = Concatenate(axis=-1)([x, y])

x = BatchNormalization()(x)
x = Activation('relu')(x)
x = GlobalAveragePooling2D()(x)
out = Dense(10, activation='softmax')(x)
model = Model(input, out)
'''

# Optimizer
# adam = Adam(lr= 0.01, beta_1= 0.9, beta_2= 0.999, epsilon= 1e-08, decay= 0.0)
sgd = SGD(lr=0.1, momentum=0.9, decay=0.0001, nesterov=True)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=sgd,
              metrics=['accuracy'])

# check the model

model.summary()

lr_decay = LearningRateScheduler(scheduler)
csv_logger = CSVLogger('SVHN_train.log')
checkpoint = ModelCheckpoint(filepath='SVHN_weights.hdf5', monitor='val_acc', verbose=0, save_best_only=True, mode='max')
callbacks_list = [lr_decay, csv_logger, checkpoint]

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=callbacks_list,
          validation_data=(x_val, y_val))

score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# visualization of the gaussian filters
# visualizeLayerOutput(model, 8, 6, 4)
# visualizeLayerOutput(model, 13, 6, 4)
# visualizeLayerOutput(model, 18, 6, 4)