# -*- coding: utf-8 -*-
"""
Created on Sat May  2 21:07:54 2015

@author: Adam Conkey

======================= PARAMETERS =======================

m_train         number of training examples
m_test          number of test examples

t_train         number of time steps in training
t_test          number of time steps in testing

n_in            number of input units
n_r             number of reservoir units
n_out           number of readout units

w_in		    weights from input to reservoir             : n_r   x n_in + 1
w_r		        weights internal to reservoir               : n_r   x n_r
w_out		    weights from reservoir to readout           : n_r   x n_out     ???
w_fb            weights from readout back into reservoir    : n_out x n_r

x_in            activations of input nodes over time        : t_train x n_in
x_r             activations of reservoir nodes over time    : t_train x n_r
x_out           activations of output nodes over time       : t_train x n_out
x_target        target activations for training             : t_train x n_out

scale_in        scale factor of input weights
scale_r		    scale factor of reservoir weights
scale_fb        scale factor of feedback weights

density_in      density coefficient for input-reservoir
density_r       density coefficient for reservoir
density_fb      density coefficient for out-reservoir

alpha	        leaking rate of reservoir units
rho		        desired spectral radius of reservoir
out_thresh      threshold for determining classification

seed 	        seed for RandomState
rs              RandomState for random number generation
==========================================================
"""
import numpy as np
from numpy.random import RandomState
from scipy import sparse
from scipy.sparse.linalg import eigs
from scipy.stats import threshold
from sklearn.linear_model import Ridge

def run_simulation():

    data = np.array()                   # THIS IS ALL TEMPORARY, will need module to actually import data
    m_train, n_in = data.shape          # and set everything appropriately
    t_train = 1000                      #
    x_in = np.array(t_train, n_in)      #
    n_out = 1                           #
    x_target = np.array(t_train, n_out) #
    # add bias node:
    x_in = np.concatenate((np.ones((t_train, 1)), x_in), axis=1)

    n_r = 100
    n_out = 1

    scale_in = 1
    scale_r = 1
    scale_fb = 1

    density_in = 1.0     # experiment with input sparsity, may get better speed without negative effect
    density_r = 0.1
    density_fb = 1.0     # experiment with feedback sparsity

    alpha = 1
    rho = 1             # NEED TO check what are good default values
    out_thresh = 0.5    # change threshold?

    seed = 123
    rs = RandomState(seed)

    # initialize input weights in range [-scale_in, scale_in]:
    w_in = initialize_weights(n_in + 1, n_r, density_in, rs, scale_in)

    # initialize reservoir in range [-scale_r, scale_r] with spectral radius rho:
    w_r = initialize_weights(n_r, n_r, density_r, rs, scale_r)
    w_r = spectral_radius(w_r, rho)

    # initialize feedback weights:
    w_fb = initialize_weights(n_out, n_r, density_fb, rs, scale_fb)

    # compute reservoir states over train duration:
    x_r = drive_network_train(n_r, t_train, x_in, x_target, w_in, w_r, w_fb, alpha)

    # compute the output weights using Ridge Regression:
    clf = Ridge()                   # play with parameters ???
    clf.fit(x_r, x_target)
    w_out = clf.coef_

    # drive with input to test accuracy (for now, just training inputs):
    x_out = drive_network_test(n_r, n_out, t_train, x_in, w_in, w_r, w_out, w_fb, out_thresh, alpha)

    # compute and output simple accuracy computation:
    print compute_accuracy(x_target, x_out)


def initialize_weights(n_rows=1, n_cols=1, density=0.1, randomstate=RandomState(1), scale=1.0):
    """
    Initialize a sparse random array of weights with dimensions
    n_rows x n_cols and specified density in range [-scale, scale].

    Keyword arguments:
    n_rows      -- number of rows (default 1)
    n_cols      -- number of columns (default 1)
    density     -- density of connections (default 0.1)
    randomstate -- RandomState object for random number generation (default RandomState(1))
    scale       -- absolute value of minimum/maximum weight value (default 1.0)
    """
    weights = sparse.rand(n_rows, n_cols, density, random_state=randomstate)
    weights = 2 * scale * weights - scale * weights.ceil() # MIGHT NOT WORK, SPARSE!!!!
    return weights

def spectral_radius(weights, spec_rad=1.0):
    """
    Scales the specified weight array to have the desired spectral radius.

    Keyword arguments:
    weights     -- weight array to scale
    spec_rad    -- desired spectral radius to scale to (default 1.0)
    """
    weights = spec_rad * (weights / max(abs(eigs(weights)[0]))) # MIGHT NOT WORK< SPARSE!!!
    return weights

def drive_network_train(n_reservoir, duration, train_data, targets,
                        w_input, w_res, w_feedback, leak_rate):
    """
    Drives the reservoir with training input and stores the reservoir states over time.

    :param n_reservoir: size of the reservoir
    :param duration: duration of the training period
    :param train_data: training input data
    :param targets: training target output values
    :param w_input: weights of connections from input layer to reservoir
    :param w_res: weights of connections in the reservoir
    :param w_feedback: weights of connections from output layer back into reservoir
    :param leak_rate: leak rate of the neurons in the reservoir
    :return: duration x n_reservoir array of reservoir activations over time
    """
    reservoir = np.zeros(duration, n_reservoir)

    for i in range(1, duration):
        reservoir[i] = np.tanh(train_data[i].dot(w_input)
                               + reservoir[i-1].dot(w_res)
                               + targets[i-1].dot(w_feedback))
        reservoir[i] = (1 - leak_rate) * reservoir[i-1] + leak_rate * reservoir[i]

    return reservoir

def drive_network_test(n_reservoir, n_output, duration, test_data, w_input,
                       w_res, w_output, w_feedback, thresh, leak_rate):
    """
    Drive the reservoir with novel input and use the trained output weights to compute output values.

    :param n_reservoir: size of the reservoir
    :param n_output: size of output layer
    :param duration: duration of testing period
    :param test_data: test input to drive the network
    :param w_input: weights for connections from input layer to reservoir
    :param w_res: weights for connections for reservoir
    :param w_output: weights for connections from reservoir to output layer
    :param w_feedback: weights for connections from output layer back to reservoir
    :param thresh: threshold determining how to round output value
    :param leak_rate: leak rate of neurons in the reservoir
    :return: duration x n_output array of computed output values over time
    """
    reservoir = np.zeros(duration, n_reservoir)
    output = np.zeros(duration, n_output)

    for i in range(1, duration):
        reservoir[i] = np.tanh(test_data[i].dot(w_input)
                               + reservoir[i-1].dot(w_res)
                               + output[i-1].dot(w_feedback))
        reservoir[i] = (1 - leak_rate) * reservoir[i-1] + leak_rate * reservoir[i]
        output[i] = np.tanh(reservoir[i].dot(w_output))

    output = threshold(output, threshmin=thresh, newval=0.0)
    output = threshold(output, threshmax=0.0, newval=1.0)

    return output

def compute_accuracy(expected, actual):
    """
    Simple computation of accuracy taking the number of correct over total.

    Keyword arguments:
    expected    -- vector of 1s and 0s, expected values
    actual      -- vector of 1s and 0s, actual values, same length as expected
    """
    n_correct = sum((expected + actual) != 1)
    total = float(len(expected))
    return n_correct / total


if __name__ == '__main__':
    run_simulation()

class EchoStateNetwork:

    "CREATE NETWORK CLASS TO PASS AROUND NETWORK AS OBJECT"

"""
STATUS: Ridge Regression done.

Need to:
    - create network object to pass around instead of a million parameters
    - create mock data set for testing and import it
    - formulate actual data set and use it
    - provide way of computing performance metrics
    - decide on sampling for performance metrics (sample only once at end of train?)
"""

"""
input sentence representation as repeated pulse trains?
"""