import scipy.io as sio
import numpy as np
import h5py
import keras

def getSVHNData():
    num_classes = 10
    svhnTrain = sio.loadmat('train_32x32.mat')
    svhnExtra = sio.loadmat('extra_32x32.mat')
    svhnTest = sio.loadmat('test_32x32.mat')

    mask1 = range(69257, 73257)
    mask2 = range(69257)
    mask3 = range(529131, 531131)
    mask4 = range(529131)

    x_train = np.concatenate([svhnTrain['X'][mask2],svhnExtra['X'][mask4]],axis=3)/255
    x_val = np.concatenate([svhnTrain['X'][mask1],svhnExtra['X'][mask3]],axis=3)/255
    x_test = svhnTest['X']/255
    y_train = np.concatenate([svhnTrain['y'][mask2],svhnExtra['y'][mask4]], axis=0) - 1
    y_val = np.concatenate([svhnTrain['y'][mask1], svhnExtra['y'][mask3]], axis=0) - 1
    y_test = svhnTest['y'] - 1

    # input image dimensions
    img_rows, img_cols = 32, 32
    print (y_val)

    x_train = np.transpose(x_train, (3, 0, 1, 2))
    x_val = np.transpose(x_val, (3, 0, 1, 2))
    x_test = np.transpose(x_test, (3, 0, 1, 2))

    # convert class vectors to binary class matrices
    y_train = y_train.ravel()
    y_val = y_val.ravel()
    y_test = y_test.ravel()
    y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.np_utils.to_categorical(y_val, num_classes)
    y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_val, y_val, x_test, y_test