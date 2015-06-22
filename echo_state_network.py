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


def run_simulation(esn, train_data):
    esn.x_in = train_data[0]
    esn.x_target = train_data[1]

    t_train, esn.n_in = esn.x_in.shape
    # add bias node:
    esn.x_in = np.concatenate((np.ones((t_train, 1)), esn.x_in), axis=1)

    # initialize input weights in range [-scale_in, scale_in]:
    esn.w_in = initialize_weights(esn.n_in + 1, esn.n_r, esn.density_in, esn.rs, esn.scale_in)

    # initialize reservoir in range [-scale_r, scale_r] with spectral radius rho:
    esn.w_r = initialize_weights(esn.n_r, esn.n_r, esn.density_r, esn.rs, esn.scale_r)
    esn.w_r = scale_spectral_radius(esn.w_r, esn.rho)

    # initialize feedback weights:
    esn.w_fb = initialize_weights(esn.n_out, esn.n_r, esn.density_fb, esn.rs, esn.scale_fb)

    # compute reservoir states over train duration:
    esn.x_r = drive_network_train(esn.n_r, esn.t_train, esn.x_in, esn.x_target, esn.w_in, esn.w_r, esn.w_fb, esn.alpha)

    # compute the output weights using Ridge Regression:
    clf = Ridge()  # play with parameters ???
    clf.fit(esn.x_r, esn.x_target)
    esn.w_out = clf.coef_

    # drive with input to test accuracy (for now, just training inputs):
    esn.x_out = drive_network_test(esn.n_r, esn.n_out, t_train, esn.x_in, esn.w_in, esn.w_r,
                                   esn.w_out, esn.w_fb, esn.out_thresh, esn.alpha)

    # compute and output simple accuracy computation:
    print compute_accuracy(esn.x_target, esn.x_out)


def initialize_weights(n_rows=1, n_cols=1, density=0.1, randomstate=RandomState(1), scale=1.0):
    """
    Initialize a sparse random array of weights with dimensions
    n_rows x n_cols and specified density in range [-scale, scale].

    The weights are generated until they achieve a spectral radius of at least 0.01;
    due to the iterative nature of scipy.sparse.linalg.eigs, values under this threshold
    are unstable and do not produce consistent results over time.

    Keyword arguments:
    n_rows      -- number of rows (default 1)
    n_cols      -- number of columns (default 1)
    density     -- density of connections (default 0.1)
    randomstate -- RandomState object for random number generation (default RandomState(1))
    scale       -- absolute value of minimum/maximum weight value (default 1.0)
    """
    while True:
        weights = sparse.rand(n_rows, n_cols, density, random_state=randomstate)
        weights = 2 * scale * weights - scale * weights.ceil()
        if max(abs(eigs(weights)[0])) >= 0.01:
            break
    return weights


def scale_spectral_radius(weights, spec_rad=1.0):
    """
    Scales the specified weight array to have the desired spectral radius.

    Keyword arguments:
    weights     -- weight array to scale
    spec_rad    -- desired spectral radius to scale to (default 1.0)
    """
    weights = spec_rad * (weights / max(abs(eigs(weights)[0])))
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
                               + reservoir[i - 1].dot(w_res)
                               + targets[i - 1].dot(w_feedback))
        reservoir[i] = (1 - leak_rate) * reservoir[i - 1] + leak_rate * reservoir[i]

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
                               + reservoir[i - 1].dot(w_res)
                               + output[i - 1].dot(w_feedback))
        reservoir[i] = (1 - leak_rate) * reservoir[i - 1] + leak_rate * reservoir[i]
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
    """
    An Echo State Network with an input layer, reservoir, and output layer with associated connections.

    Attributes:
        n_in            number of input units
        n_r             number of reservoir units
        n_out           number of readout units

        w_in		    weights from input to reservoir             : n_r   x n_in + 1
        w_r		        weights internal to reservoir               : n_r   x n_r
        w_out		    weights from reservoir to readout           : n_r   x n_out
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
    """

    def __init__(self, n_in, n_r, n_out, w_in, w_r, w_out, w_fb, x_in, x_r, x_out,
                 x_target, scale_in, scale_r, scale_fb, density_in, density_r,
                 density_fb, alpha, rho, out_thresh, seed, rs):
        self.n_in = n_in
        self.n_r = n_r
        self.n_out = n_out
        self.w_in = w_in
        self.w_r = w_r
        self.w_out = w_out
        self.w_fb = w_fb
        self.x_in = x_in
        self.x_r = x_r
        self.x_out = x_out
        self.x_target = x_target
        self.scale_in = scale_in
        self.scale_r = scale_r
        self.scale_fb = scale_fb
        self.density_in = density_in
        self.density_r = density_r
        self.density_fb = density_fb
        self.alpha = alpha
        self.rho = rho
        self.out_thresh = out_thresh
        self.seed = seed
        self.rs = rs


"""
STATUS: Ridge Regression done.

Need to:
    - create mock data set for testing and import it
    - formulate actual data set and use it
    - provide way of computing performance metrics
    - decide on sampling for performance metrics (sample only once at end of train?)
"""

"""
input sentence representation as repeated pulse trains?
"""
