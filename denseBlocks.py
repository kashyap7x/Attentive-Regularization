from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Concatenate, Input, BatchNormalization, Activation, AveragePooling2D
from keras import regularizers, constraints
from keras import backend as K
from layer import Target2D
from keras.models import Model

def DenseNet(nb_dense_block = 3, growth_rate=48, reduction=0.5, dropout_rate=0.0, weight_decay=1e-4, classes=10, include_target='false', l2=0.01, l2_buildup = 1, weights_path=None):
    '''Instantiate the DenseNet 121 architecture,
        # Arguments
            nb_dense_block: number of dense blocks to add to end
            growth_rate: number of filters to add per dense block
            nb_filter: initial number of filters
            reduction: reduction factor of transition blocks.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            classes: optional number of classes to classify images
            weights_path: path to pre-trained weights
        # Returns
            A Keras model instance.
    '''
    eps = 1.1e-5

    # compute compression factor
    compression = 1.0 - reduction

    # Handle Dimension Ordering for different backends
    global concat_axis
    if K.image_dim_ordering() == 'tf':
      concat_axis = 3
      img_input = Input(shape=(32, 32, 3))
    else:
      concat_axis = 1
      img_input = Input(shape=(3, 32, 32))

    # From architecture for ImageNet (Table 1 in the paper)
    nb_filter = growth_rate * 2
    nb_layers = [6,6,6]

    # Initial convolution
    x = Conv2D(nb_filter, kernel_size=(3, 3), padding='same', use_bias=False)(img_input)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx+2
        x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay, include_target = include_target, l2 = l2, l2_buildup = l2_buildup)

        # Add transition_block
        x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    final_stage = stage + 1
    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay, include_target = include_target, l2 = l2, l2_buildup = l2_buildup)

    x = BatchNormalization(epsilon=eps, axis=concat_axis)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    x = Dense(classes)(x)
    x = Activation('softmax')(x)

    model = Model(img_input, x, name='densenet')

    if weights_path is not None:
      model.load_weights(weights_path)

    return model


def conv_block(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4, include_target='false', l2=0.01, l2_buildup = 1):
    '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
        # Arguments
            x: input tensor
            stage: index for dense block
            branch: layer index within each dense block
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''
    eps = 1.1e-5

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4
    x = BatchNormalization(epsilon=eps, axis=concat_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(inter_channel, (1, 1), use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = DenseTarget2D(x, growth_rate=nb_filter, include_target=include_target, l2=l2)
    l2 *= l2_buildup

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_filter: number of filters
            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''

    eps = 1.1e-5

    x = BatchNormalization(epsilon=eps, axis=concat_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(int(nb_filter * compression), (1, 1), use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2))(x)

    return x


def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True, include_target='false', l2=0.01, l2_buildup = 1):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
    '''

    eps = 1.1e-5
    concat_feat = x

    for i in range(nb_layers):
        branch = i+1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay, include_target = include_target, l2 = l2, l2_buildup = l2_buildup)
        concat_feat = Concatenate(axis=concat_axis)([concat_feat, x])

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter

def DenseTarget2D(x, growth_rate, include_target = 'false', attention_function='cauchy', l2=0.01):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    y = Conv2D(growth_rate, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    if include_target == 'true':
        y = Target2D(attention_function=attention_function, sig1_regularizer=regularizers.l2(l2),sig2_regularizer=regularizers.l2(l2))(y)
    return y