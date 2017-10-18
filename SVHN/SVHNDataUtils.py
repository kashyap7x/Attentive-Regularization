import scipy.io as sio
import numpy as np
import h5py
import keras

def getSVHNData():
    num_classes = 10
    svhnTrain = sio.loadmat('train_32x32.mat')
    svhnTest = sio.loadmat('test_32x32.mat')

    xTrain = svhnTrain['X']
    xTest = svhnTest['X']
    yTrain = svhnTrain['y']
    yTest = svhnTest['y']

    # input image dimensions
    img_rows, img_cols = 32, 32

    x_train = xTrain
    y_train = yTrain
    x_test = xTest
    y_test = yTest

    x_train = np.transpose(x_train, (3, 0, 1, 2))
    x_test = np.transpose(x_test, (3, 0, 1, 2))

    '''
    trainMean = np.mean(x_train, axis=0)
    print(trainMean)
    trainSigma = np.std(x_train, axis=0)
    print(trainSigma)
    '''

    trainMeanFile = h5py.File('SVHN_preprocessing.h5', 'r')
    #trainMeanFile.create_dataset('train_mean', data=trainMean)
    #trainMeanFile.create_dataset('train_sigma', data=trainSigma)
    trainMean = np.array(trainMeanFile['train_mean'])
    trainSigma = np.array(trainMeanFile['train_sigma'])
    trainMeanFile.close()

    x_train = (x_train - trainMean)/trainSigma
    x_test = (x_test - trainMean)/trainSigma

    # convert class vectors to binary class matrices
    y_train -= 1
    y_test -= 1
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test