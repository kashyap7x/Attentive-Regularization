from __future__ import print_function
import sys
sys.path.append('..')
from keras import backend as K
from keras.layers import Dense, Flatten, Dropout, Input, Lambda
from keras.optimizers import Adam
from keras import regularizers
from keras.models import Model
from keras_vggface.vggface import VGGFace
from keras import metrics
from layer import Target2D
from visualization import visualizeLayerOutput
from lfwDataUtils import getLFWData

def get_abs_diff(vects):
    x, y = vects
    return K.abs(x - y)

def abs_diff_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1

batchSize = 16
numEpoch = 3

input_shape = (224, 224, 3)
left_input = Input(input_shape)
right_input = Input(input_shape)

# Baseline model
base = VGGFace(include_top=True, input_shape=(224, 224, 3), pooling='None')
'''
x = base.get_layer('conv4_1').output
x = Target2D(attention_function='cauchy', sig1_regularizer=regularizers.l2(0.01), sig2_regularizer=regularizers.l2(0.01))(x)
x = base.get_layer('conv4_2')(x)
x = Target2D(attention_function='cauchy', sig1_regularizer=regularizers.l2(0.01), sig2_regularizer=regularizers.l2(0.01))(x)
x = base.get_layer('conv4_3')(x)
x = Target2D(attention_function='cauchy', sig1_regularizer=regularizers.l2(0.01), sig2_regularizer=regularizers.l2(0.01))(x)
x = base.get_layer('pool4')(x)
x = base.get_layer('conv5_1')(x)
x = Target2D(attention_function='cauchy', sig1_regularizer=regularizers.l2(0.01), sig2_regularizer=regularizers.l2(0.01))(x)
x = base.get_layer('conv5_2')(x)
x = Target2D(attention_function='cauchy', sig1_regularizer=regularizers.l2(0.01), sig2_regularizer=regularizers.l2(0.01))(x)
x = base.get_layer('conv5_3')(x)
x = Target2D(attention_function='cauchy', sig1_regularizer=regularizers.l2(0.01), sig2_regularizer=regularizers.l2(0.01))(x)
x = base.get_layer('pool5')(x)
x = base.get_layer('flatten')(x)
x = base.get_layer('fc6')(x)
x = base.get_layer('fc6/relu')(x)
x = base.get_layer('fc7')(x)
out = base.get_layer('fc7/relu')(x)
'''
out = base.get_layer('fc7/relu').output
model = Model(base.input, out)

# Freezing pretrained layers
for layer in model.layers:
   layer.trainable = False
'''
model.layers[12].trainable = True
model.layers[14].trainable = True
model.layers[16].trainable = True
model.layers[19].trainable = True
model.layers[21].trainable = True
model.layers[23].trainable = True
'''
encoded_l = model(left_input)
encoded_r = model(right_input)

both = Lambda(get_abs_diff, output_shape = abs_diff_output_shape)([encoded_l,encoded_r])
prediction = Dense(1,activation='sigmoid')(both)
siamese_net = Model(input=[left_input,right_input],output=prediction)

siamese_net.summary()

# Optimizer
adam = Adam(lr= 0.001, beta_1= 0.9, beta_2= 0.999, epsilon= 1e-08, decay= 0.0)

# Compile and train
siamese_net.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=[metrics.binary_accuracy])

f = 0
trainList = [i for i in range(10) if i is not f]
for e in range(numEpoch):
    print("Epoch %d/%d" % (e+1,numEpoch))
    for i in trainList:
        X1_train, X2_train, Y_train = getLFWData(i)
        siamese_net.fit([X1_train, X2_train], Y_train,
                        batch_size=batchSize,
                        epochs=1,
                        shuffle=True)
    X1_test, X2_test, y_test = getLFWData(f)
    print(siamese_net.evaluate([X1_test, X2_test], y_test,
                         batch_size=batchSize))
'''
visualizeLayerOutput(model, layerNum = 12)
visualizeLayerOutput(model, layerNum = 14)
visualizeLayerOutput(model, layerNum = 16)
visualizeLayerOutput(model, layerNum = 19)
visualizeLayerOutput(model, layerNum = 21)
visualizeLayerOutput(model, layerNum = 23)
'''