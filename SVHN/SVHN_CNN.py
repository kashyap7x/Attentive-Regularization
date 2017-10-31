from __future__ import print_function
import sys
sys.path.append('..')
import keras
from keras.optimizers import Adam, SGD
from keras.layers import Dense, Conv2D, Concatenate, MaxPooling2D, GlobalAveragePooling2D, Input, BatchNormalization, Activation
from keras.models import Model
#from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, CSVLogger, ModelCheckpoint
from denseBlocks import DenseTarget2D
from denseBlocks import DenseNet, wideResNet
from SVHNDataUtils import getSVHNData
from visualization import *
np.random.seed(1337)

batch_size = 64
epochs = 20

def scheduler(epoch):
    if epoch==10 or epoch==15:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr*.1)
        print("lr changed to {}".format(lr*.1))
    return K.get_value(model.optimizer.lr)

x_train, y_train, x_val, y_val, x_test, y_test = getSVHNData()
#datagen = ImageDataGenerator(rescale= 1. / 255)

#model = wideResNet(k=8, dropout=0.4, include_target='false')
model = DenseNet(growth_rate=12, reduction=0.5, dropout_rate=0, include_target='false')

# Optimizer
# adam = Adam(lr= 0.01, beta_1= 0.9, beta_2= 0.999, epsilon= 1e-08, decay= 0.0)
sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)

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
'''
model.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size),
                    steps_per_epoch=x_train.shape[0] // batch_size,
                    epochs=epochs,
                    verbose=1,
                    callbacks=callbacks_list,
                    validation_data=(x_val, y_val))
'''

score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# visualization of the gaussian filters
# visualizeLayerOutput(model, 8, 6, 4)
# visualizeLayerOutput(model, 13, 6, 4)
# visualizeLayerOutput(model, 18, 6, 4)