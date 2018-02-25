from keras import backend as K
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


def make_mosaic(imgs, nrows, ncols, getRangeX, getRangeY, border=1):
    """
    Given a set of images with all the same shape, makes a mosaic for visualizing the target layer
    :param imgs: images
    :param nrows: # rows
    :param ncols: # cols
    :param getRangeX: X range for the ROI box
    :param getRangeY: Y range for the ROI box
    :param border: number of pixels between images
    :return:
    """

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
        im = (imgs[i].copy()*255).astype('uint8')
        #im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        #print(im)
        cv2.rectangle(im, (getRangeX[0,i,0], getRangeY[0,i,0]), (getRangeX[1,i,0], getRangeY[1,i,0]), 200, 1)
        mosaic[row * paddedh:row * paddedh + imshape[0],
        col * paddedw:col * paddedw + imshape[1]] = im
    return mosaic


def nice_imshow(ax, data, vmin=None, vmax=None, cmap=cm.gray, fName='fig.png'):
    """
    Wrapper around plt.imshow. Currently disabled so as to save images directly
    :param ax:
    :param data:
    :param vmin:
    :param vmax:
    :param cmap:
    :param fName:
    :return:
    """
    '''
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    plt.colorbar(im, cax=cax)
    plt.axis('off')
    '''
    plt.imsave(fName,data, cmap=cmap)


def visualizeLayerOutput(model, layerNum=12, ySize=16, xSize=32):
    """
    Plots AR window
    :param model: keras model
    :param layerNum: target layer number
    :param ySize: grid Y size
    :param xSize: grid X size
    :return:
    """
    getFunction = K.eval(model.layers[layerNum].function)
    getRangeX = K.eval(model.layers[layerNum].rangeX)
    getRangeY = K.eval(model.layers[layerNum].rangeY)
    output_image = np.array(getFunction)
    output_image = np.squeeze(output_image)
    output_image = np.transpose(output_image, (2, 1, 0))
    plt.figure(figsize=(ySize * 0.75, xSize * 0.75))
    nice_imshow(plt.gca(), make_mosaic(output_image, ySize, xSize, getRangeX, getRangeY), cmap=cm.gray, fName=str(layerNum) + '.png')


def visualizeLayerOutput1D(model, layerNum=3, fName='fig.png'):
    """
    1 dimensional version of visualizeLayerOutput, corresponding to the AR1D layer
    :param model: keras model
    :param layerNum: AR1D layer number
    :param fName: filename to save plot
    :return:
    """
    getFunction = K.eval(model.layers[layerNum].function)
    # print(np.shape(getFunction))
    output_image = np.array(getFunction)
    plt.imshow(output_image, cmap='gray')
    plt.savefig(fName, bbox_inches='tight')