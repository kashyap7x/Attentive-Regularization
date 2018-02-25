from keras import activations, initializers, regularizers, constraints
from keras.utils import conv_utils
from keras.engine import Layer, InputSpec
from keras.layers import Conv2D, Concatenate, BatchNormalization, Activation
from keras import backend as K
import numpy as np
import tensorflow as tf


class AR1D(Layer):
    '''
	One-dimensional attentive regularization layer. Use after Convolution1D.

	# Usage
		x = AR1D()(x)
		model.add(AR1D())

	# Arguments
		attention_function: name of the attention function used. 'gaussian' or 'cauchy'.
		input_length: Number of time/sequence steps in the input.
		input_dim: Number of channels/dimensions in the input.
		mu_constraint: instance of the [constraints](../constraints.md) module
			(eg. maxnorm, nonneg), applied to the attention function means.
		sig_constraint: constraint applied to the attention function standard deviations.
		mu_regularizer: instance of [WeightRegularizer](../regularizers.md)
			(eg. L1 or L2 regularization), applied to the attention function means.
		sig_regularizer: regularizer applied to the attention function standard deviations.

	# Input shape
		3D tensor with shape: (samples, input_length, input_dim).

	# Output shape
		Same shape as input.
	'''

    def __init__(self, attention_function='gaussian',
                 mu_regularizer=None, sig_regularizer=None,
                 input_length=None, input_dim=None, **kwargs):

        if attention_function not in {'gaussian', 'cauchy'}:
            raise Exception('Invalid attention function', attention_function)
        self.attention_function = attention_function

        self.mu_regularizer = regularizers.get(mu_regularizer)
        self.sig_regularizer = regularizers.get(sig_regularizer)

        self.input_spec = [InputSpec(ndim=3)]
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)

        super(AR1D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        input_length = input_shape[1]
        input_dim = input_shape[2]

        mu_init = np.zeros((input_dim, 1), K.floatx())
        sig_init = np.ones((input_dim, 1), K.floatx())
        base = np.tile(np.arange(input_length, dtype=K.floatx()), (input_dim, 1))

        self.mu = tf.Variable(mu_init)
        self.sig = tf.Variable(sig_init)
        if self.mu_regularizer is not None:
            self.add_loss(self.mu_regularizer(self.mu))
        if self.sig_regularizer is not None:
            self.add_loss(self.sig_regularizer(self.sig))
        self.trainable_weights = [self.mu, self.sig]

        self.base = tf.Variable(base)
        self.mu_tiled = tf.tile(self.mu, (1, input_length))
        self.sig_tiled = tf.tile(self.sig, (1, input_length))
        self.numerator = self.base - self.mu_tiled * input_length
        self.denominator = self.sig_tiled * input_length / 2

        if self.attention_function == 'gaussian':
            self.function = tf.transpose(K.exp(-(self.numerator) ** 2. / (2 * (self.denominator) ** 2)))
        else:
            self.function = tf.transpose(1 / (1 + ((self.numerator) / self.denominator) ** 2))

    def call(self, x, mask=None):
        return x * self.function


class AR2D(Layer):
    '''
	Two-dimensional attentive regularization layer. Use after Convolution2D.

	# Usage
		x = AR2D()(x)
		model.add(AR2D())

	# Arguments
		attention_function: name of the attention function used. 'gaussian' or 'cauchy'.
		input_length: input image side in pixels.
		input_dim: number of channels/dimensions in the input.
		mu1_constraint: instance of the [constraints](../constraints.md) module
			(eg. maxnorm, nonneg), applied to the attention function means.
		sig1_constraint: constraint applied to the attention function standard deviations.
		mu1_regularizer: instance of [WeightRegularizer](../regularizers.md)
			(eg. L1 or L2 regularization), applied to the attention function means.
		sig1_regularizer: regularizer applied to the attention function standard deviations.
		mu2_constraint: constraint applied to the attention function means.
		sig2_constraint: constraint applied to the attention function standard deviations.
		mu2_regularizer: regularizer applied to the attention function means.
		sig2_regularizer: regularizer applied to the attention function standard deviations.

	# Input shape
		4D tensor with shape: (samples, input_length, input_length, input_dim).

	# Output shape
		Same shape as input.
	'''

    def __init__(self, attention_function='gaussian',
                 mu1_constraint=None, sig1_constraint=None,
                 mu1_regularizer=None, sig1_regularizer=None,
                 mu2_constraint=None, sig2_constraint=None,
                 mu2_regularizer=None, sig2_regularizer=None,
                 input_length=None, input_dim=None, **kwargs):

        if attention_function not in {'gaussian', 'cauchy'}:
            raise Exception('Invalid attention function', attention_function)
        self.attention_function = attention_function

        self.mu1_regularizer = regularizers.get(mu1_regularizer)
        self.sig1_regularizer = regularizers.get(sig1_regularizer)
        self.mu1_constraint = constraints.get(mu1_constraint)
        self.sig1_constraint = constraints.get(sig1_constraint)
        self.mu2_regularizer = regularizers.get(mu2_regularizer)
        self.sig2_regularizer = regularizers.get(sig2_regularizer)
        self.mu2_constraint = constraints.get(mu2_constraint)
        self.sig2_constraint = constraints.get(sig2_constraint)

        self.input_spec = [InputSpec(ndim=4)]
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)

        super(AR2D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        input_length = input_shape[1]
        input_dim = input_shape[3]

        mu_init = np.ones((input_dim, 1), K.floatx()) / 2
        sig_init = np.ones((input_dim, 1), K.floatx())
        base = np.tile(np.arange(input_length, dtype=K.floatx()), (input_dim, 1))

        self.mu1 = tf.Variable(mu_init)
        self.sig1 = tf.Variable(sig_init)
        self.mu2 = tf.Variable(mu_init)
        self.sig2 = tf.Variable(sig_init)
        if self.mu1_regularizer is not None:
            self.add_loss(self.mu1_regularizer(self.mu1))
        if self.sig1_regularizer is not None:
            self.add_loss(self.sig1_regularizer(self.sig1))
        if self.mu2_regularizer is not None:
            self.add_loss(self.mu2_regularizer(self.mu2))
        if self.sig2_regularizer is not None:
            self.add_loss(self.sig2_regularizer(self.sig2))
        self.trainable_weights = [self.mu1, self.sig1, self.mu2, self.sig2]

        self.base = tf.Variable(base)
        self.mu1_tiled = tf.tile(self.mu1, (1, input_length))
        self.sig1_tiled = tf.tile(self.sig1, (1, input_length))
        self.numerator1 = self.base - self.mu1_tiled * input_length
        self.denominator1 = self.sig1_tiled * input_length / 2

        if self.attention_function == 'gaussian':
            self.function1 = tf.transpose(
                tf.tile(
                    tf.expand_dims(tf.transpose(K.exp(-(self.numerator1) ** 2. / (2 * (self.denominator1) ** 2))),
                                   2),
                    (1, 1, input_length)), perm=[0, 2, 1])

        else:
            self.function1 = tf.transpose(
                tf.tile(
                    tf.expand_dims(tf.transpose(1 / (1 + ((self.numerator1) / self.denominator1) ** 2)), 2),
                    (1, 1, input_length)), perm=[0, 2, 1])

        self.mu2_tiled = tf.tile(self.mu2, (1, input_length))
        self.sig2_tiled = tf.tile(self.sig2, (1, input_length))
        self.numerator2 = tf.transpose(self.base - self.mu2_tiled * input_length)
        self.denominator2 = tf.transpose(self.sig2_tiled * input_length / 2)

        if self.attention_function == 'gaussian':
            self.function2 = tf.transpose(
                tf.tile(
                    tf.expand_dims((tf.transpose(K.exp(-(self.numerator2) ** 2. / (2 * (self.denominator2) ** 2)))),
                                   2),
                    (1, 1, input_length,)), perm=[2, 1, 0])

        else:
            self.function2 = tf.transpose(
                tf.tile(
                    tf.expand_dims((tf.transpose(1 / (1 + ((self.numerator2) / self.denominator2) ** 2))), 2),
                    (1, 1, input_length,)), perm=[2, 1, 0])

        self.function = self.function1 * self.function2

    def call(self, x, mask=None):
        return x * self.function


'''
def custom_mu_init(shape, dtype=K.floatx()):
    return K.ones(shape, dtype=dtype) / 2
'''


class Target2D(Layer):
    """2 Dimensional Target Layer. Combines Conv2D and AR2D, with a faster convolution implementation.

    This layer creates a convolution kernel that is convolved
    with the layer input to produce a tensor of outputs.
    If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`,
    it is applied to the outputs as well.

    # Arguments
        rank: An integer, the rank of the convolution,
            e.g. "2" for 2D convolution.
        filters: Integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution).
        kernel_size: An integer or tuple/list of n integers, specifying the
            dimensions of the convolution window.
        strides: An integer or tuple/list of n integers,
            specifying the strides of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: One of `"valid"` or `"same"` (case-insensitive).
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, ..., channels)` while `channels_first` corresponds to
            inputs with shape `(batch, channels, ...)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        dilation_rate: An integer or tuple/list of n integers, specifying
            the dilation rate to use for dilated convolution.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any `strides` value != 1.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to the kernel matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        attention_function: name of the attention function used. 'gaussian' or 'cauchy'.
		input_length: input image side in pixels.
		input_dim: number of channels/dimensions in the input.
		mu1_constraint: instance of the [constraints](../constraints.md) module
			(eg. maxnorm, nonneg), applied to the attention function means.
		sig1_constraint: constraint applied to the attention function standard deviations.
		mu1_regularizer: instance of [WeightRegularizer](../regularizers.md)
			(eg. L1 or L2 regularization), applied to the attention function means.
		sig1_regularizer: regularizer applied to the attention function standard deviations.
		mu2_constraint: constraint applied to the attention function means.
		sig2_constraint: constraint applied to the attention function standard deviations.
		mu2_regularizer: regularizer applied to the attention function means.
		sig2_regularizer: regularizer applied to the attention function standard deviations.
    """

    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 preslice=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 attention_function='gaussian',
                 mu1_regularizer=None,
                 sig1_regularizer=None,
                 mu2_regularizer=None,
                 sig2_regularizer=None,
                 mu1_constraint=None,
                 sig1_constraint=None,
                 mu2_constraint=None,
                 sig2_constraint=None,
                 input_length=None,
                 input_dim=None,
                 **kwargs):
        super(Target2D, self).__init__(**kwargs)
        self.rank = 2
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, self.rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, self.rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, self.rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.preslice = preslice
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=self.rank + 2)

        if attention_function not in {'gaussian', 'cauchy'}:
            raise Exception('Invalid attention function', attention_function)
        self.attention_function = attention_function
        self.mu1_regularizer = regularizers.get(mu1_regularizer)
        self.sig1_regularizer = regularizers.get(sig1_regularizer)
        self.mu1_constraint = constraints.get(mu1_constraint)
        self.sig1_constraint = constraints.get(sig1_constraint)
        self.mu2_regularizer = regularizers.get(mu2_regularizer)
        self.sig2_regularizer = regularizers.get(sig2_regularizer)
        self.mu2_constraint = constraints.get(mu2_constraint)
        self.sig2_constraint = constraints.get(sig2_constraint)

        self.input_spec = [InputSpec(ndim=4)]
        self.input_dim = input_dim
        self.input_length = input_length

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        # Target addition
        self.input_dim = input_shape[channel_axis]
        input_dim = self.input_dim

        kernel_shape = self.kernel_size + (input_dim, self.filters)
        self.input_length = input_shape[2]
        input_length = self.input_length
        input_dim = kernel_shape[-1]

        mu_init = np.ones((input_dim, 1), K.floatx()) / 2
        sig_init = np.ones((input_dim, 1), K.floatx())
        base = np.tile(np.arange(input_length, dtype=K.floatx()), (input_dim, 1))

        self.mu1 = tf.Variable(mu_init, name='mu1')
        self.sig1 = tf.Variable(sig_init, name='sig1')
        self.mu2 = tf.Variable(mu_init, name='mu2')
        self.sig2 = tf.Variable(sig_init, name='sig2')

        if self.mu1_regularizer is not None:
            self.add_loss(self.mu1_regularizer(self.mu1))
        if self.sig1_regularizer is not None:
            self.add_loss(self.sig1_regularizer(self.sig1))
        if self.mu2_regularizer is not None:
            self.add_loss(self.mu2_regularizer(self.mu2))
        if self.sig2_regularizer is not None:
            self.add_loss(self.sig2_regularizer(self.sig2))

        self.trainable_weights = [self.mu1, self.sig1, self.mu2, self.sig2]
        input_dim = self.input_dim

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})

        '''
        #self.mu1 = self.add_weight(shape=mu_init.shape,
                                   initializer=custom_mu_init,
                                   name='mu1',
                                   regularizer=self.mu1_regularizer,
                                   constraint=self.mu1_constraint)

        #self.mu2 = self.add_weight(shape=mu_init.shape,
                                   initializer=custom_mu_init,
                                   name='mu2',
                                   regularizer=self.mu2_regularizer,
                                   constraint=self.mu2_constraint)

        #self.sig1 = self.add_weight(shape=sig_init.shape,
                                    initializer='Ones',
                                    name='sig1',
                                    regularizer=self.sig1_regularizer,
                                    constraint=self.sig1_constraint)

        #self.sig2 = self.add_weight(shape=sig_init.shape,
                                    initializer='Ones',
                                    name='sig2',
                                    regularizer=self.sig2_regularizer,
                                    constraint=self.sig2_constraint)
        '''
        self.mu1 = tf.clip_by_value(self.mu1, 1/input_length, 1-(1/input_length))
        self.mu2 = tf.clip_by_value(self.mu2, 1/input_length, 1-(1/input_length))
        self.sig1 = tf.clip_by_value(self.sig1, 3/(2*input_length), 1)
        self.sig2 = tf.clip_by_value(self.sig2, 3/(2*input_length), 1)


        self.base = tf.Variable(base)
        self.mu1_tiled = tf.tile(self.mu1, (1, input_length))
        self.sig1_tiled = tf.tile(self.sig1, (1, input_length))
        self.numerator1 = self.base - self.mu1_tiled * input_length
        self.denominator1 = self.sig1_tiled * input_length / 2
        self.mu2_tiled = tf.tile(self.mu2, (1, input_length))
        self.sig2_tiled = tf.tile(self.sig2, (1, input_length))
        self.numerator2 = tf.transpose(self.base - self.mu2_tiled * input_length)
        self.denominator2 = tf.transpose(self.sig2_tiled * input_length / 2)

        if self.attention_function == 'gaussian':
            self.function1 = tf.transpose(
                tf.tile(
                    tf.expand_dims(tf.transpose(K.exp(-(self.numerator1) ** 2. / (2 * (self.denominator1) ** 2))),
                                   2),
                    (1, 1, input_length)), perm=[0, 2, 1])

        else:
            self.function1 = tf.transpose(
                tf.tile(
                    tf.expand_dims(tf.transpose(1 / (1 + ((self.numerator1) / self.denominator1) ** 2)), 2),
                    (1, 1, input_length)), perm=[0, 2, 1])

        if self.attention_function == 'gaussian':
            self.function2 = tf.transpose(
                tf.tile(
                    tf.expand_dims((tf.transpose(K.exp(-(self.numerator2) ** 2. / (2 * (self.denominator2) ** 2)))),
                                   2),
                    (1, 1, input_length,)), perm=[2, 1, 0])

        else:
            self.function2 = tf.transpose(
                tf.tile(
                    tf.expand_dims((tf.transpose(1 / (1 + ((self.numerator2) / self.denominator2) ** 2))), 2),
                    (1, 1, input_length,)), perm=[2, 1, 0])

        rangeX = [(self.mu1 - self.sig1/np.sqrt(2)) * input_length, (self.mu1 + self.sig1/np.sqrt(2)) * input_length + 1]
        rangeY = [(self.mu2 - self.sig2/np.sqrt(2)) * input_length, (self.mu2 + self.sig2/np.sqrt(2)) * input_length + 1]

        self.rangeY = tf.to_int32(tf.clip_by_value(rangeY, 0, input_length - 1))
        self.rangeX = tf.to_int32(tf.clip_by_value(rangeX, 0, input_length - 1))

        self.function = self.function1 * self.function2

        self.built = True

    def call(self, inputs):

        if self.preslice:
            outputHeight = self.input_length
            numFilters = self.kernel.get_shape().as_list()[-1]
            kernList = tf.unstack(self.kernel, axis=3)
            outputs = []
            for i in range(numFilters):
                kernel = tf.expand_dims(kernList[i], 3)
                output = K.conv2d(
                    inputs[:, self.rangeY[0,i,0]:self.rangeY[1,i,0], self.rangeX[0,i,0]:self.rangeX[1,i,0], :],
                    kernel,
                    strides=self.strides,
                    padding=self.padding,
                    data_format=self.data_format,
                    dilation_rate=self.dilation_rate)

                paddings = [[0, 0], [self.rangeY[0,i,0], outputHeight - self.rangeY[1,i,0]],
                            [self.rangeX[0,i,0], outputHeight - self.rangeX[1,i,0]], [0, 0]]

                outputs.append(tf.pad(output, paddings, "CONSTANT"))

            outputs = tf.concat(outputs, axis=3)
        else:
            outputs = K.conv2d(
                    inputs,
                    self.kernel,
                    strides=self.strides,
                    padding=self.padding,
                    data_format=self.data_format,
                    dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs * self.function

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (self.filters,)
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0], self.filters) + tuple(new_space)

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'preslice': self.preslice,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'attention_function': self.attention_function,
            'mu1_regularizer': regularizers.serialize(self.mu1_regularizer),
            'sig1_regularizer': regularizers.serialize(self.sig1_regularizer),
            'mu2_regularizer': regularizers.serialize(self.mu2_regularizer),
            'sig2_regularizer': regularizers.serialize(self.sig2_regularizer),
            'mu1_constraint': constraints.serialize(self.mu1_constraint),
            'sig1_constraint': constraints.serialize(self.sig1_constraint),
            'mu2_constraint': constraints.serialize(self.mu2_constraint),
            'sig2_constraint': constraints.serialize(self.sig2_constraint)
        }
        base_config = super(Target2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SpatialTransformer(Layer):
    """Spatial Transformer Layer
    Implements a spatial transformer layer as described in [1]_.
    Borrowed from [2]_:
    downsample_fator : float
        A value of 1 will keep the orignal size of the image.
        Values larger than 1 will down sample the image. Values below 1 will
        upsample the image.
        example image: height= 100, width = 200
        downsample_factor = 2
        output image will then be 50, 100
    References
    ----------
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py
    .. [3]  https://github.com/EderSantana/seya/blob/keras1/seya/layers/attention.py
    """

    def __init__(self,
                 localization_net,
                 output_size,
                 **kwargs):
        self.locnet = localization_net
        self.output_size = output_size
        super(SpatialTransformer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.locnet.build(input_shape)
        self.trainable_weights = self.locnet.trainable_weights
        #self.regularizers = self.locnet.regularizers //NOT SUER ABOUT THIS, THERE IS NO MORE SUCH PARAMETR AT self.locnet
        self.constraints = self.locnet.constraints

    def compute_output_shape(self, input_shape):
        output_size = self.output_size
        return (None,
                int(output_size[0]),
                int(output_size[1]),
                int(input_shape[-1]))

    def call(self, X, mask=None):
        affine_transformation = self.locnet.call(X)
        output = self._transform(affine_transformation, X, self.output_size)
        return output

    def _repeat(self, x, num_repeats):
        ones = tf.ones((1, num_repeats), dtype='int32')
        x = tf.reshape(x, shape=(-1,1))
        x = tf.matmul(x, ones)
        return tf.reshape(x, [-1])

    def _interpolate(self, image, x, y, output_size):
        batch_size = tf.shape(image)[0]
        height = tf.shape(image)[1]
        width = tf.shape(image)[2]
        num_channels = tf.shape(image)[3]

        x = tf.cast(x , dtype='float32')
        y = tf.cast(y , dtype='float32')

        height_float = tf.cast(height, dtype='float32')
        width_float = tf.cast(width, dtype='float32')

        output_height = output_size[0]
        output_width  = output_size[1]

        x = .5*(x + 1.0)*(width_float)
        y = .5*(y + 1.0)*(height_float)

        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        max_y = tf.cast(height - 1, dtype='int32')
        max_x = tf.cast(width - 1,  dtype='int32')
        zero = tf.zeros([], dtype='int32')

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)

        flat_image_dimensions = width*height
        pixels_batch = tf.range(batch_size)*flat_image_dimensions
        flat_output_dimensions = output_height*output_width
        base = self._repeat(pixels_batch, flat_output_dimensions)
        base_y0 = base + y0*width
        base_y1 = base + y1*width
        indices_a = base_y0 + x0
        indices_b = base_y1 + x0
        indices_c = base_y0 + x1
        indices_d = base_y1 + x1

        flat_image = tf.reshape(image, shape=(-1, num_channels))
        flat_image = tf.cast(flat_image, dtype='float32')
        pixel_values_a = tf.gather(flat_image, indices_a)
        pixel_values_b = tf.gather(flat_image, indices_b)
        pixel_values_c = tf.gather(flat_image, indices_c)
        pixel_values_d = tf.gather(flat_image, indices_d)

        x0 = tf.cast(x0, 'float32')
        x1 = tf.cast(x1, 'float32')
        y0 = tf.cast(y0, 'float32')
        y1 = tf.cast(y1, 'float32')

        area_a = tf.expand_dims(((x1 - x) * (y1 - y)), 1)
        area_b = tf.expand_dims(((x1 - x) * (y - y0)), 1)
        area_c = tf.expand_dims(((x - x0) * (y1 - y)), 1)
        area_d = tf.expand_dims(((x - x0) * (y - y0)), 1)
        output = tf.add_n([area_a*pixel_values_a,
                           area_b*pixel_values_b,
                           area_c*pixel_values_c,
                           area_d*pixel_values_d])
        return output

    def _meshgrid(self, height, width):
        x_linspace = tf.linspace(-1., 1., width)
        y_linspace = tf.linspace(-1., 1., height)
        x_coordinates, y_coordinates = tf.meshgrid(x_linspace, y_linspace)
        x_coordinates = tf.reshape(x_coordinates, [-1])
        y_coordinates = tf.reshape(y_coordinates, [-1])
        ones = tf.ones_like(x_coordinates)
        indices_grid = tf.concat([x_coordinates, y_coordinates, ones], 0)
        return indices_grid

    def _transform(self, affine_transformation, input_shape, output_size):
        batch_size = tf.shape(input_shape)[0]
        height = tf.shape(input_shape)[1]
        width = tf.shape(input_shape)[2]
        num_channels = tf.shape(input_shape)[3]

        affine_transformation = tf.reshape(affine_transformation, shape=(batch_size,2,3))

        affine_transformation = tf.reshape(affine_transformation, (-1, 2, 3))
        affine_transformation = tf.cast(affine_transformation, 'float32')

        width = tf.cast(width, dtype='float32')
        height = tf.cast(height, dtype='float32')
        output_height = output_size[0]
        output_width = output_size[1]
        indices_grid = self._meshgrid(output_height, output_width)
        indices_grid = tf.expand_dims(indices_grid, 0)
        indices_grid = tf.reshape(indices_grid, [-1]) # flatten?

        indices_grid = tf.tile(indices_grid, tf.stack([batch_size]))
        indices_grid = tf.reshape(indices_grid, (batch_size, 3, -1))

        transformed_grid = tf.matmul(affine_transformation, indices_grid)
        x_s = tf.slice(transformed_grid, [0, 0, 0], [-1, 1, -1])
        y_s = tf.slice(transformed_grid, [0, 1, 0], [-1, 1, -1])
        x_s_flatten = tf.reshape(x_s, [-1])
        y_s_flatten = tf.reshape(y_s, [-1])

        transformed_image = self._interpolate(input_shape,
                                                x_s_flatten,
                                                y_s_flatten,
                                                output_size)

        transformed_image = tf.reshape(transformed_image, shape=(batch_size,
                                                                output_height,
                                                                output_width,
                                                                num_channels))
        return transformed_image