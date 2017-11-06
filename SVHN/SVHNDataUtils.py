import scipy.io as sio
import numpy as np
import h5py
import keras
import random
random.seed(1337)

def getSVHNData(full=False):
    train_data = sio.loadmat('train_32x32.mat', variable_names='X').get('X')
    train_labels = sio.loadmat('train_32x32.mat', variable_names='y').get('y')
    test_data = sio.loadmat('test_32x32.mat', variable_names='X').get('X')
    test_labels = sio.loadmat('test_32x32.mat', variable_names='y').get('y')

    n_labels = 10

    train_labels[train_labels == n_labels] = 0
    test_labels[test_labels == n_labels] = 0

    if full:
        extra_data = sio.loadmat('extra_32x32.mat', variable_names='X').get('X')
        extra_labels = sio.loadmat('extra_32x32.mat', variable_names='y').get('y')

        extra_labels[extra_labels == n_labels] = 0

        valid_index = []
        valid_index2 = []
        train_index = []
        train_index2 = []
        for i in np.arange(n_labels):
            valid_index.extend(np.where(train_labels[:, 0] == (i))[0][:400].tolist())
            train_index.extend(np.where(train_labels[:, 0] == (i))[0][400:].tolist())
            valid_index2.extend(np.where(extra_labels[:, 0] == (i))[0][:200].tolist())
            train_index2.extend(np.where(extra_labels[:, 0] == (i))[0][200:].tolist())

        '''
        valid_index = list(range(69257, 73257))
        train_index = list(range(69257))
        valid_index2 = list(range(529131, 531131))
        train_index2 = list(range(529131))

        random.shuffle(valid_index)
        random.shuffle(train_index)
        random.shuffle(valid_index2)
        random.shuffle(train_index2)
        '''

        x_val = np.concatenate((extra_data[:, :, :, valid_index2], train_data[:, :, :, valid_index]),
                                    axis=3).transpose((3, 0, 1, 2))
        y_val = np.concatenate((extra_labels[valid_index2, :], train_labels[valid_index, :]), axis=0)[:, 0]
        x_train = np.concatenate((extra_data[:, :, :, train_index2], train_data[:, :, :, train_index]),
                                      axis=3).transpose((3, 0, 1, 2))
        y_train = np.concatenate((extra_labels[train_index2, :], train_labels[train_index, :]), axis=0)[:, 0]

        '''
        print (np.bincount(np.squeeze(y_train))/529131)
        print (np.bincount(np.squeeze(y_val))/6000)
        print (np.bincount(np.squeeze(y_test))/26032)
        '''
    else:
        x_train = train_data.transpose((3, 0, 1, 2))
        y_train = train_labels[:, 0]
        x_val = test_data.transpose((3, 0, 1, 2))
        y_val = test_labels[:, 0]

    x_test = test_data.transpose((3, 0, 1, 2))
    y_test = test_labels[:, 0]

    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)
    print(x_test.shape, y_test.shape)

    # convert class vectors to binary class matrices
    y_train = keras.utils.np_utils.to_categorical(y_train, n_labels)
    y_val = keras.utils.np_utils.to_categorical(y_val, n_labels)
    y_test = keras.utils.np_utils.to_categorical(y_test, n_labels)

    return x_train, y_train, x_val, y_val, x_test, y_test