from __future__ import print_function
import sys
sys.path.append('..')
import keras
from keras.optimizers import Adam
from denseBlocks import DenseNet, wideResNet
from SVHNDataUtils import getSVHNData
from visualization import *

batch_size = 128
epochs = 20

x_train, y_train, x_test, y_test = getSVHNData()

model = wideResNet(k=8, dropout=0.4, include_target='false')
#model = DenseNet(growth_rate=48, reduction=0.5, dropout_rate=0.0, include_target='false')

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