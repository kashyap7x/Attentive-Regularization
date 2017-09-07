from __future__ import print_function
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.models import Model
from keras_vggface.vggface import VGGFace
from AUDetectDataUtils import getCKData

batchSize = 16
numEpoch = 20
nb_classes = 17

# Load and check data
X_train, y_train, X_val, y_val, X_test, y_test = getCKData()
print ('Train data shape: ', X_train.shape)
print ('Train labels shape: ', y_train.shape)
print ('Validation data shape: ', X_val.shape)
print ('Validation labels shape: ', y_val.shape)
print ('Test data shape: ', X_test.shape)
print ('Test labels shape: ', y_test.shape)

# Baseline model
base = VGGFace(include_top=False, input_shape=(224, 224, 3), pooling='None')
last_layer = base.get_layer('pool5').output
x = Flatten(name='flatten1')(last_layer)
x = Dense(1024, activation='relu', name='fc6')(x)
out = Dense(nb_classes, activation='sigmoid', name='fc7')(x)
model = Model(base.input, out)
model.summary()

# Freezing pretrained layers
for layer in model.layers[:19]:
   layer.trainable = False
for layer in model.layers[19:]:
   layer.trainable = True

# Optimizer
adam = Adam(lr= 0.001, beta_1= 0.9, beta_2= 0.999, epsilon= 1e-08, decay= 0.0)

# Compile and train
model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

model.fit(X_train, y_train,
	batch_size=batchSize,
	epochs=numEpoch,
	validation_data=(X_val, y_val),
	shuffle=True)

# Check performance on test data
preds = model.predict(X_test)
preds[preds>=0.5] = 1
preds[preds<0.5] = 0
print(preds - y_test)