from __future__ import absolute_import
import theano
import theano.tensor as T
import numpy as np
from six.moves import range

if theano.config.floatX == 'float64':
    epsilon = 1.0e-9
else:
    epsilon = 1.0e-7


def mean_squared_error(y_true, y_pred):
    return T.sqr(y_pred - y_true).mean(axis=-1)


def mean_absolute_error(y_true, y_pred):
    return T.abs_(y_pred - y_true).mean(axis=-1)


def mean_absolute_percentage_error(y_true, y_pred):
    return T.abs_((y_true - y_pred) / T.clip(T.abs_(y_true), epsilon, np.inf)).mean(axis=-1) * 100.


def mean_squared_logarithmic_error(y_true, y_pred):
    return T.sqr(T.log(T.clip(y_pred, epsilon, np.inf) + 1.) - T.log(T.clip(y_true, epsilon, np.inf) + 1.)).mean(axis=-1)


def squared_hinge(y_true, y_pred):
    return T.sqr(T.maximum(1. - y_true * y_pred, 0.)).mean(axis=-1)


def hinge(y_true, y_pred):
    return T.maximum(1. - y_true * y_pred, 0.).mean(axis=-1)


def categorical_crossentropy(y_true, y_pred):
    '''Expects a binary class matrix instead of a vector of scalar classes
    '''
    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
    # scale preds so that the class probas of each sample sum to 1
    y_pred /= y_pred.sum(axis=-1, keepdims=True)
    cce = T.nnet.categorical_crossentropy(y_pred, y_true)
    return cce


def binary_crossentropy(y_true, y_pred):
    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
    bce = T.nnet.binary_crossentropy(y_pred, y_true).mean(axis=-1)
    return bce

def GAN_generator_loss(y_true, y_pred):
    #y_true should be a two column vector with second column all ones
    return T.log( epsilon + (y_true - y_true*y_pred).sum(axis=-1) ).mean(axis=-1)

def GAN_generator_loss2(y_true, y_pred):
    #y_true should be a two column vector with second column all ones
    return -1.*T.log( epsilon + (y_true*y_pred).sum(axis=-1) ).mean(axis=-1)

def my_bce(y_true, y_pred):
    return -1.*T.log(epsilon + (y_true*y_pred).sum(axis=-1) ).mean(axis=-1)

def my_bce_pos(y_true, y_pred):
    return T.log(epsilon + (y_true*y_pred).sum(axis=-1) ).mean(axis=-1)

def GAN_discriminator_loss(y_true, y_pred):
    return -1.*T.log( epsilon + (y_true*y_pred).sum(axis=-1) ).mean(axis=-1)

def poisson_loss(y_true, y_pred):
    return T.mean(y_pred - y_true * T.log(y_pred + epsilon), axis=-1)

def weighted_mse(y_true, y_pred):
    n = T.shape(y_true)[-1]/2
    return (T.sqr(y_pred - y_true[:,:n])*y_true[:,n:]).mean(axis=-1)

def ytrue_weighted_mse(y_true, y_pred):
    d = T.sqr(y_pred - y_true)
    e = (y_true+0.1) * d
    return e.mean(axis=-1)

def ytrue_weighted_mae(y_true, y_pred):
    d = T.abs_(y_pred - y_true)
    e = (y_true+0.1) * d
    return e.mean(axis=-1)

def weighted_bce(y_true, y_pred):
    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
    n = T.shape(y_true)[-1]/2
    e = y_true[:,n:]*( y_true[:,:n]*T.log(y_pred) + (1 - y_true[:,:n])*T.log(1-y_pred) ) 
    return -1*e.mean(axis=-1)

# aliases
mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error

from .utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'objective')
