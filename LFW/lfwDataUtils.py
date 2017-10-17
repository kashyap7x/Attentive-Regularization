import numpy as np
import cv2
import h5py
from keras import backend as K

def preprocess(x):
    data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if data_format == 'channels_first':
        x = x[:, ::-1, ...]
        temp1 = x[:, 0, :, :] - 93.5940
        temp2 = x[:, 1, :, :] - 104.7624
        temp3 = x[:, 2, :, :] - 129.1863
        x[:, 0, :, :] = temp3
        x[:, 1, :, :] = temp2
        x[:, 2, :, :] = temp1
    else:
        x = x[..., ::-1]
        temp1 = x[..., 0] - 93.5940
        temp2 = x[..., 1] - 104.7624
        temp3 = x[..., 2] - 129.1863
        x[..., 0] = temp3
        x[..., 1] = temp2
        x[..., 2] = temp1
    return x

def getLFWData(f):
    '''
    X1_list = np.zeros([5400, 96, 96, 3])
    X2_list = np.zeros([5400, 96, 96, 3])
    Y_list = np.zeros(5400)
    curLength = 0
    trainList = [i for i in range(10) if i is not f]
    for f in trainList:
        dataFile = h5py.File('fold' + str(f) + '.hdf5', 'r')
        X1 = np.array(dataFile['X1'])
        X2 = np.array(dataFile['X2'])
        Y = np.array(dataFile['Y'])
        X1_list[curLength:curLength + 600] = X1
        X2_list[curLength:curLength + 600] = X2
        Y_list[curLength:curLength + 600] = Y
        curLength += 600

    X1_train = preprocess(X1_list).astype('float32')
    X2_train = preprocess(X2_list).astype('float32')
    y_train = Y_list
    '''

    dataFile = h5py.File('fold' + str(f) + '.hdf5', 'r')
    X1 = np.array(dataFile['X1'])
    X2 = np.array(dataFile['X2'])
    Y = np.array(dataFile['Y'])
    X1_test = preprocess(X1).astype('float32')
    X2_test = preprocess(X2).astype('float32')
    y_test = Y
    return X1_test, X2_test, y_test