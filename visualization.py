from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None, fName='fig.png'):
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
    plt.savefig(fName, bbox_inches='tight')

def visualizeLayerOutput(model, layerNum=12, ySize=16, xSize=32):
    # Displays the selected input in visual form and
    # outputs of the output layer for a single input with a given input in both
    # pictorial and numerical form

    getFunction = K.eval(model.layers[layerNum].function)
    output_image = np.array(getFunction)
    output_image = np.squeeze(output_image)
    output_image = np.transpose(output_image, (2, 1, 0))
    plt.figure(figsize=(ySize * 0.75, xSize * 0.75))
    plt.title('Target2D')
    nice_imshow(plt.gca(), make_mosaic(output_image, ySize, xSize), cmap=cm.binary, fName=str(layerNum) + '.png')

def visualizeLayerOutput1D(model, layerNum=3, fName='fig.png'):
    # Displays the selected input in visual form and
    # outputs of the output layer for a single input with a given input in both
    # pictorial and numerical form

    getFunction = K.eval(model.layers[layerNum].function)
    # print(np.shape(getFunction))
    output_image = np.array(getFunction)
    plt.imshow(output_image, cmap='gray')
    plt.savefig(fName, bbox_inches='tight')