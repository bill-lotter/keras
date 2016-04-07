from __future__ import absolute_import
import numpy as np
from . import backend as K


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


def mean_absolute_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)


def mean_absolute_percentage_error(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true), K.epsilon(), np.inf))
    return 100. * K.mean(diff, axis=-1)


def mean_squared_logarithmic_error(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), np.inf) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), np.inf) + 1.)
    return K.mean(K.square(first_log - second_log), axis=-1)


def squared_hinge(y_true, y_pred):
    return K.mean(K.square(K.maximum(1. - y_true * y_pred, 0.)), axis=-1)


def hinge(y_true, y_pred):
    return K.mean(K.maximum(1. - y_true * y_pred, 0.), axis=-1)


def categorical_crossentropy(y_true, y_pred):
    '''Expects a binary class matrix instead of a vector of scalar classes.
    '''
    return K.categorical_crossentropy(y_pred, y_true)


def binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)


def poisson(y_true, y_pred):
    return K.mean(y_pred - y_true * K.log(y_pred + K.epsilon()), axis=-1)


def GAN_generator_loss(y_true, y_pred):
    #y_true should be a two column vector with second column all ones
    return K.log( epsilon + (y_true - y_true*y_pred).sum(axis=-1) ).mean(axis=-1)

def GAN_generator_loss2(y_true, y_pred):
    #y_true should be a two column vector with second column all ones
    return -1.*K.log( epsilon + (y_true*y_pred).sum(axis=-1) ).mean(axis=-1)

def my_bce(y_true, y_pred):
    return -1.*K.log(epsilon + (y_true*y_pred).sum(axis=-1) ).mean(axis=-1)

def my_bce_pos(y_true, y_pred):
    return K.log(epsilon + (y_true*y_pred).sum(axis=-1) ).mean(axis=-1)

def GAN_discriminator_loss(y_true, y_pred):
    return -1.*K.log( epsilon + (y_true*y_pred).sum(axis=-1) ).mean(axis=-1)

def cosine_proximity(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return -K.mean(y_true * y_pred, axis=-1)


def weighted_mse(y_true, y_pred):
    n = K.shape(y_true)[-1]/2
    return (K.sqr(y_pred - y_true[:,:n])*y_true[:,n:]).mean(axis=-1)

def ytrue_weighted_mse(y_true, y_pred):
    d = K.sqr(y_pred - y_true)
    e = (y_true+0.1) * d
    return e.mean(axis=-1)

def ytrue_weighted_mae(y_true, y_pred):
    d = K.abs_(y_pred - y_true)
    e = (y_true+0.1) * d
    return e.mean(axis=-1)

def weighted_bce(y_true, y_pred):
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    n = K.shape(y_true)[-1]/2
    e = y_true[:,n:]*( y_true[:,:n]*K.log(y_pred) + (1 - y_true[:,:n])*K.log(1-y_pred) )
    return -1*e.mean(axis=-1)

# y_true is mask
def masked_sum(y_true, y_pred):
    return -1.0* K.mean(y_true * y_pred, axis=-1)

# aliases
mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
cosine = cosine_proximity

from .utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'objective')
