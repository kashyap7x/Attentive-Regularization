import numpy as np
import cv2
import h5py

def getCKData(num_training=493, num_validation=50, num_test=50):
	# Subsample loaded data for validation and test

	dataFile = h5py.File('ckplus_faces.hdf5', 'r')

	keys = list(dataFile.keys())
	images = dataFile[keys[0]]
	images = np.array(images).astype(np.uint8)
	for image in images:
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	labelFile = h5py.File('FACS_labels_reduced.h5', 'r')

	keys = list(labelFile.keys())
	labels = labelFile[keys[0]]

	# Split the data
	mask = range(num_training)
	X_train = images[mask]
	y_train = labels[mask]
	mask = range(num_training, num_training + num_validation)
	X_val = images[mask]
	y_val = labels[mask]
	mask = range(num_training + num_validation, num_training + num_validation + num_test)
	X_test = images[mask]
	y_test = labels[mask]

	# Normalize the data: subtract the mean image and place in (0,1)
	X_train = X_train.astype('float32')
	X_val = X_val.astype('float32')
	X_test = X_test.astype('float32')

	mean_image = np.mean(X_train, axis=0)
	X_train -= mean_image
	X_val -= mean_image
	X_test -= mean_image

	X_train /= 255
	X_val /= 255
	X_test /= 255

	return X_train, y_train, X_val, y_val, X_test, y_test