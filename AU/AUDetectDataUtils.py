import numpy as np
import cv2
import h5py
from keras import backend as K
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

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


def make_mosaic(imgs, nrows, ncols, border=1):
	# Given a set of images with all the same shape, makes a
	# mosaic with nrows and ncols
	nimgs = imgs.shape[0]
	imshape = imgs.shape[1:]

	mosaic = np.ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
							ncols * imshape[1] + (ncols - 1) * border),
						   dtype=np.float32)

	paddedh = imshape[0] + border
	paddedw = imshape[1] + border
	for i in range(nimgs):
		row = int(np.floor(i / ncols))
		col = i % ncols

		mosaic[row * paddedh:row * paddedh + imshape[0],
		col * paddedw:col * paddedw + imshape[1]] = imgs[i]
	return mosaic

def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
	# Wrapper around plt.imshow
	if cmap is None:
		cmap = cm.jet
	if vmin is None:
		vmin = data.min()
	if vmax is None:
		vmax = data.max()
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
	plt.colorbar(im, cax=cax)
	plt.show()

def visualizeLayerOutput(model, layerNum = 10):
	# Displays the selected input in visual form and
	# outputs of the output layer for a single input with a given input in both
	# pictorial and numerical form

	getFunction = K.eval(model.layers[layerNum].function)
	output_image = np.array(getFunction)
	output_image = np.squeeze(output_image)
	output_image = np.transpose(output_image, (2, 1, 0))
	plt.figure(figsize=(20, 20))
	plt.title('Target2D')
	nice_imshow(plt.gca(), make_mosaic(output_image, 16, 16), cmap=cm.binary)