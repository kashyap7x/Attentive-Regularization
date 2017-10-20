from __future__ import print_function
import sys
sys.path.append('..')
import keras
from keras.optimizers import Adam, SGD
from denseBlocks import DenseNet, wideResNet
from SVHNDataUtils import getSVHNData
from visualization import *
from keras.callbacks import LearningRateScheduler, CSVLogger, ModelCheckpoint

batch_size = 128
epochs = 40

def scheduler(epoch):
    if epoch==20 or epoch==30:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr*.1)
        print("lr changed to {}".format(lr*.1))
    return K.get_value(model.optimizer.lr)

x_train, y_train, x_test, y_test = getSVHNData()

model = wideResNet(k=8, dropout=0.4, include_target='false')
#model = DenseNet(growth_rate=48, reduction=0.5, dropout_rate=0.2, include_target='false')

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
          validation_data=(x_test, y_test))

# visualization of the gaussian filters
# visualizeLayerOutput(model, 10, 4, 4)
# visualizeLayerOutput(model, 16, 4, 4)
# visualizeLayerOutput(model, 22, 4, 4)