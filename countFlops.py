from keras import backend as K


def countFlops(model):
    """
    Counts the total additions and multiplications in a model
    :param model: keras model
    :return: #FLOPS
    """
    flops = 0
    for layer in model.layers:
        type = (layer.name[0:4])
        delFlops = 0
        if type == 'conv':
            shape = K.eval(layer.kernel).shape
            delFlops = shape[0] * shape[1] * shape[2] * shape[3] * layer.output_shape[1] * layer.output_shape[2]
        elif type == 'targ':
            shape = K.eval(layer.kernel).shape
            getRangeX = K.eval(layer.rangeX)
            getRangeY = K.eval(layer.rangeY)
            for i in range(shape[3]):
                delFlops += shape[0] * shape[1] * shape[2] * (getRangeX[1, i, 0]-getRangeX[0, i, 0]-1) * (getRangeY[1, i, 0]-getRangeY[0, i, 0]-1)
        elif type == 'batc':
            if (len(layer.output_shape)) == 2:
                delFlops = layer.output_shape[1]
            else:
                delFlops = layer.output_shape[1] * layer.output_shape[2] * layer.output_shape[3]
        elif type == 'dens':
            shape = K.eval(layer.kernel).shape
            delFlops = (shape[0] + 1)* shape[1]
        flops += delFlops
        print (type, delFlops, flops)
    return(flops)