from .. import regularizers, constraints
from ..engine import Layer, InputSpec
from .. import backend as K
import numpy as np
import theano

class AttentiveRegularization1D(Layer):
    '''
	One-dimensional attentive regularization layer. Use after Convolution1D.
	    
    # Usage
    	x = AttentiveRegularization1D()(x)
    	model.add(AttentiveRegularization1D())
    
    # Arguments
        attention_function: name of the attention function used. 'gaussian' or 'cauchy'.
        input_length: Number of timesteps in the input.
        input_dim: Number of channels/dimensions in the input.
        mu_constraint: instance of the [constraints](../constraints.md) module 
            (eg. maxnorm, nonneg), applied to the attention function means.
        sig_constraint: instance of the [constraints](../constraints.md) module 
            (eg. maxnorm, nonneg), applied to the attention function standard deviations.
        mu_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the attention function means.
        sig_regularizer: instance of [WeightRegularizer](../regularizers.md), 
            applied to the attention function standard deviations.
        
    # Input shape
        3D tensor with shape: (samples, input_length, input_dim).

    # Output shape
        Same shape as input.
    '''
    
    def __init__(self, attention_function = 'gaussian', 
                 mu_constraint=None, sig_constraint=None,
                 mu_regularizer=None, sig_regularizer=None,
                 input_length=None, input_dim=None, **kwargs):
        
        if attention_function not in {'gaussian', 'cauchy'}:
            raise Exception('Invalid attention function', attention_function)
        self.attention_function = attention_function
        
        self.mu_regularizer = regularizers.get(mu_regularizer)
        self.sig_regularizer = regularizers.get(sig_regularizer)
        self.mu_constraint = constraints.get(mu_constraint)
        self.sig_constraint = constraints.get(sig_constraint)
                
        self.input_spec = [InputSpec(ndim=3)]
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        
        super(AttentiveRegularization1D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        input_length = input_shape[1]
        input_dim = input_shape[2]
        
        #mu_init = np.arange(input_length, dtype=theano.config.floatX) - (input_length/2 * (1 + (np.random.rand(input_dim,1)*2-1).astype(theano.config.floatX)))
        #sig_init = np.tile(np.ones((input_dim,1),theano.config.floatX)*input_length/2, (1,input_length))
        
        mu_init = np.zeros((input_dim,1),theano.config.floatX)
        #mu_init = (np.random.randn(input_dim,1).astype(theano.config.floatX) + 1)/2
        sig_init = np.ones((input_dim,1),theano.config.floatX)
        base = np.tile(np.arange(input_length, dtype=theano.config.floatX), (input_dim,1))
        
        self.mu = theano.shared(mu_init)
        self.sig = theano.shared(sig_init)
        self.trainable_weights = [self.mu, self.sig]

        self.base = theano.shared(base)
        self.non_trainable_weights = [self.base]
        
        self.mu_tiled = theano.tensor.tile(self.mu, (1,input_length))
        self.sig_tiled = theano.tensor.tile(self.sig, (1,input_length))
        self.numerator = self.base - self.mu_tiled * input_length
        self.denominator = self.sig_tiled * input_length/2
        
        if self.attention_function == 'gaussian':
        	self.function = (K.exp(-(self.numerator)**2./(2*(self.denominator)**2))).T
        else:
        	self.function = (1/(1+((self.numerator)/self.denominator)**2)).T
        	
        self.regularizers = []
        if self.mu_regularizer:
            self.mu_regularizer.set_param(self.mu)
            self.regularizers.append(self.mu_regularizer)

        if self.sig_regularizer:
            self.sig_regularizer.set_param(self.sig)
            self.regularizers.append(self.sig_regularizer)
            
        self.constraints = {}
        if self.mu_constraint:
            self.constraints[self.mu] = self.mu_constraint
        if self.sig_constraint:
            self.constraints[self.sig] = self.sig_constraint

    def call(self, x, mask=None):
        return x * self.function
    
    
class AttentiveRegularization2D(Layer):
    '''
    Two-dimensional attentive regularization layer. Use after Convolution2D.
	    
    # Usage
    	x = AttentiveRegularization2D()(x)
    	model.add(AttentiveRegularization2D())
    
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
    
    def __init__(self, attention_function = 'gaussian', 
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
        
        super(AttentiveRegularization2D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        input_length = input_shape[1]
        input_dim = input_shape[3]
        
        mu_init = np.ones((input_dim,1),theano.config.floatX)/2
        sig_init = np.ones((input_dim,1),theano.config.floatX)
        base = np.tile(np.arange(input_length, dtype=theano.config.floatX), (input_dim,1))
        
        self.mu1 = theano.shared(mu_init)
        self.sig1 = theano.shared(sig_init)
        self.mu2 = theano.shared(mu_init)
        self.sig2 = theano.shared(sig_init)
        self.trainable_weights = [self.mu1, self.sig1, self.mu2, self.sig2]

        self.base = theano.shared(base)
        self.non_trainable_weights = [self.base]
        
        self.mu1_tiled = theano.tensor.tile(self.mu1, (1,input_length))
        self.sig1_tiled = theano.tensor.tile(self.sig1, (1,input_length))
        self.numerator1 = self.base - self.mu1_tiled * input_length
        self.denominator1 = self.sig1_tiled * input_length/2
                
        if self.attention_function == 'gaussian':
        	self.function1 = theano.tensor.tile(K.exp(-(self.numerator1)**2./(2*(self.denominator1)**2)), (input_length,1,1)).T
        else:
        	self.function1 = theano.tensor.tile(1/(1+((self.numerator1)/self.denominator1)**2), (input_length,1,1)).T
        	
        self.mu2_tiled = theano.tensor.tile(self.mu2, (1,input_length))
        self.sig2_tiled = theano.tensor.tile(self.sig2, (1,input_length))
        self.numerator2 = (self.base - self.mu2_tiled * input_length).T
        self.denominator2 = (self.sig2_tiled * input_length/2).T
        self.function2 = theano.tensor.tile((K.exp(-(self.numerator2)**2./(2*(self.denominator2)**2))).T, (input_length,1,1))
        
        if self.attention_function == 'gaussian':
        	self.function2 = theano.tensor.tile((K.exp(-(self.numerator2)**2./(2*(self.denominator2)**2))).T, (input_length,1,1))
        else:
        	self.function2 = theano.tensor.tile((1/(1+((self.numerator2)/self.denominator2)**2)).T, (input_length,1,1))
        	
        self.function = np.swapaxes(self.function1 * self.function2, 1,2)
        
        self.regularizers = []
        if self.mu1_regularizer:
            self.mu1_regularizer.set_param(self.mu1)
            self.regularizers.append(self.mu1_regularizer)

        if self.sig1_regularizer:
            self.sig1_regularizer.set_param(self.sig1)
            self.regularizers.append(self.sig1_regularizer)
            
        if self.mu2_regularizer:
            self.mu2_regularizer.set_param(self.mu2)
            self.regularizers.append(self.mu2_regularizer)

        if self.sig2_regularizer:
            self.sig2_regularizer.set_param(self.sig2)
            self.regularizers.append(self.sig2_regularizer)
            
        self.constraints = {}
        if self.mu1_constraint:
            self.constraints[self.mu1] = self.mu1_constraint
        if self.sig1_constraint:
            self.constraints[self.sig1] = self.sig1_constraint
        if self.mu2_constraint:
            self.constraints[self.mu2] = self.mu2_constraint
        if self.sig2_constraint:
            self.constraints[self.sig2] = self.sig2_constraint

    def call(self, x, mask=None):
        return x * self.function
        
