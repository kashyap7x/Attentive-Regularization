from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Concatenate, Input, BatchNormalization, Activation, AveragePooling2D, Add, Dropout, Flatten, Lambda
from keras import regularizers, constraints
from keras import backend as K
from layer import AR2D, Target2D
from keras.models import Model

def DenseTarget2D(x, growth_rate, weight_decay=1e-4, include_target = 'false', attention_function='cauchy', l2=0.00001, use_bias=False):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    if include_target == 'false':
        x = Conv2D(growth_rate, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same', use_bias=use_bias,kernel_regularizer=regularizers.l2(weight_decay))(x)
    elif include_target == 'preslice':
        x = Target2D(growth_rate, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same', use_bias=use_bias, preslice=True, kernel_regularizer=regularizers.l2(weight_decay), attention_function=attention_function, sig1_regularizer=regularizers.l2(l2),sig2_regularizer=regularizers.l2(l2))(x)
    else:
        x = Target2D(growth_rate, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same', use_bias=use_bias,kernel_regularizer=regularizers.l2(weight_decay), attention_function=attention_function, sig1_regularizer=regularizers.l2(l2),sig2_regularizer=regularizers.l2(l2))(x)
    return x

def DenseNet(in_shape=(32, 32, 3), nb_dense_block = 3, lay_per_block = 6, growth_rate=48, bottleneck=True, reduction=0.5, dropout_rate=0.0, weight_decay=1e-4, classes=10, include_target='false', l2=0.01, l2_buildup = 1, weights_path=None):
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
    concat_axis = 3
    img_input = Input(shape=in_shape)

    # From architecture for ImageNet (Table 1 in the paper)
    nb_filter = growth_rate * 2
    nb_layers = [lay_per_block,lay_per_block,lay_per_block]

    # Initial convolution
    x = Lambda(lambda x: x / 255)(img_input)
    x = Conv2D(nb_filter, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same', use_bias=False,kernel_regularizer=regularizers.l2(weight_decay))(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx+2
        x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, bottleneck=bottleneck, dropout_rate=dropout_rate, weight_decay=weight_decay, include_target = include_target, l2 = l2)
        l2 *= l2_buildup

        # Add transition_block
        x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    final_stage = stage + 1
    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, bottleneck=bottleneck, dropout_rate=dropout_rate, weight_decay=weight_decay, include_target = include_target, l2 = l2)

    x = BatchNormalization(epsilon=eps, axis=concat_axis)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    x = Dense(classes)(x)
    x = Activation('softmax')(x)

    model = Model(img_input, x, name='densenet')

    if weights_path is not None:
      model.load_weights(weights_path)

    return model


def dense_block(x, stage, nb_layers, nb_filter, growth_rate, bottleneck=True, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True, include_target='false', l2=0.01):
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

    concat_feat = x

    for i in range(nb_layers):
        branch = i+1
        x = conv_block(concat_feat, stage, branch, growth_rate, bottleneck, dropout_rate, weight_decay, include_target = include_target, l2 = l2)
        if bottleneck:
            concat_feat = Concatenate(axis=concat_axis)([concat_feat, x])

            if grow_nb_filters:
                nb_filter += growth_rate

        else:
            concat_feat = Concatenate(axis=concat_axis)([concat_feat, x])
            x = conv_block(concat_feat, stage, branch, growth_rate, bottleneck, dropout_rate, weight_decay,
                           include_target=include_target, l2=l2)
            concat_feat = Concatenate(axis=concat_axis)([concat_feat, x])

            if grow_nb_filters:
                nb_filter += growth_rate*2

    return concat_feat, nb_filter


def conv_block(x, stage, branch, nb_filter, bottleneck=True, dropout_rate=None, weight_decay=1e-4, include_target='false', l2=0.0001):
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

    if bottleneck:
    # 1x1 Convolution (Bottleneck layer)
        inter_channel = nb_filter * 2
        x = BatchNormalization(epsilon=eps, axis=concat_axis)(x)
        x = Activation('relu')(x)
        x = Conv2D(inter_channel, (1, 1),kernel_initializer='he_normal', padding='same', use_bias=False,kernel_regularizer=regularizers.l2(weight_decay))(x)

        if dropout_rate:
            x = Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = DenseTarget2D(x, growth_rate=nb_filter, weight_decay=weight_decay, include_target=include_target, l2=l2)

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
    x = Conv2D(int(nb_filter * compression), (1, 1),kernel_initializer='he_normal', padding='same', use_bias=False,kernel_regularizer=regularizers.l2(weight_decay))(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2))(x)

    return x

def wideResNet(k, dropout, include_target='false', l2=0.01, l2_buildup = 1):
    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        img_input = Input(shape=(3, 32, 32))
    else:
        img_input = Input(shape=(32, 32, 3))

    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    x = Lambda(lambda x: x / 255)(img_input)
    x = Conv2D(16 * k, (3, 3), padding='same')(x)

    x = conv_block_resnet(x, [16 * k, 16 * k], strides=(1, 1), dropout=dropout, include_target = include_target, l2 = l2, l2_buildup = l2_buildup)
    x = identity_block(x, [16 * k, 16 * k], dropout=dropout, include_target = include_target, l2 = l2, l2_buildup = l2_buildup)

    x = conv_block_resnet(x, [32 * k, 32 * k], dropout=dropout, include_target = include_target, l2 = l2, l2_buildup = l2_buildup)
    x = identity_block(x, [32 * k, 32 * k], dropout=dropout, include_target = include_target, l2 = l2, l2_buildup = l2_buildup)

    x = conv_block_resnet(x, [64 * k, 64 * k], dropout=dropout, include_target = include_target, l2 = l2, l2_buildup = l2_buildup)
    x = identity_block(x, [64 * k, 64 * k], dropout=dropout, include_target = include_target, l2 = l2, l2_buildup = l2_buildup)

    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = AveragePooling2D((8, 8))(x)
    x = Flatten()(x)
    x = Dense(10, activation='softmax')(x)

    model = Model(img_input, x)

    return model

def identity_block(input_tensor, filters, dropout, include_target='false', l2=0.01, l2_buildup = 1):
    '''The identity_block is the block that has no conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    nb_filter1, nb_filter2 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    x = DenseTarget2D(input_tensor, growth_rate=nb_filter1, include_target=include_target, l2=l2, use_bias=True)
    l2 *= l2_buildup

    if dropout>0:
        x = Dropout(dropout)(x)

    x = DenseTarget2D(x, growth_rate=nb_filter2, include_target=include_target, l2=l2, use_bias=True)
    l2 *= l2_buildup

    x = Add()([x, input_tensor])
    return x


def conv_block_resnet(input_tensor, filters, strides=(2, 2), dropout=0, include_target='false', l2=0.01, l2_buildup = 1):
    '''conv_block is the block that has a conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    '''
    nb_filter1, nb_filter2 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    x = BatchNormalization(axis=bn_axis)(input_tensor)
    x = Activation('relu')(x)

    shortcut = Conv2D(nb_filter2, (1, 1), strides=strides)(x)
    x = Conv2D(nb_filter1, (3, 3), strides=strides, padding='same')(x)
    if include_target == 'true':
        x = AR2D(attention_function='cauchy', sig1_regularizer=regularizers.l2(l2),sig2_regularizer=regularizers.l2(l2))(x)
    l2 *= l2_buildup

    if dropout>0:
        x = Dropout(dropout)(x)

    x = DenseTarget2D(x, growth_rate=nb_filter2, include_target=include_target, l2=l2, use_bias=True)
    l2 *= l2_buildup

    x = Add()([x, shortcut])
    return x