# -*- coding: utf-8 -*-
"""
Created on Sat May  2 21:07:54 2015

@author: Adam Conkey

"""
import numpy as np
from numpy.random import RandomState
from scipy import sparse
from scipy.sparse.linalg import eigs
from sklearn.linear_model import Ridge
from sklearn.cross_validation import train_test_split
import random

def run_simulation(esn):
    """
    Train the output weights of the provided Echo State Network with the network's associated
    training set and then drive the network with its associated testing set.

    :param esn: Echo State Network to be trained and tested
    :return: computed accuracy as ratio of correct classifications / total test examples
    """
    # compute reservoir states over train duration:
    drive_network_train(esn)

    # compute the output weights using Ridge Regression:
    clf = Ridge()  # play with parameters ???
    clf.fit(esn.x_r, esn.x_target_train)
    esn.w_out = clf.coef_

    # drive with test input:
    drive_network_test(esn)

    # compute and return simple accuracy computation:
    return compute_accuracy(esn.x_target_test, esn.x_out)

def drive_network_train(esn):
    """
    Drives the reservoir with training input and stores the reservoir states over time.

    :param esn: Echo State Network to train with
    """
    for instance in esn.train_data:
        x_in = np.vstack((np.zeros((1, esn.n_in)), instance[0]))
        x_target = np.vstack((np.zeros((1, esn.n_out)),
                              np.tile(instance[1], (instance[0].shape[0]))))  # repeat for each time step
        x_instance = np.zeros((x_in.shape[0], esn.n_r))
        for i in range(1, x_in.shape[0] + 1):
            x_instance[i] = np.tanh((x_in[i] * esn.w_in)
                                    + (x_instance[i - 1] * esn.w_r)
                                    + (x_target[i - 1] * esn.w_fb))
            x_instance[i] = (1 - esn.alpha) * x_instance[i - 1] + esn.alpha * x_instance[i]
        np.vstack((esn.x_r, x_instance[1:]))  # add instance activation except initial zero activation
        np.vstack((esn.x_target_train, x_target[1:]))  # same with targets


def drive_network_test(esn):
    """
    Drive the reservoir with novel input and use the trained output weights to compute output values.

    :param esn: Echo State Network to test with
    """
    targets = np.zeros((esn.test_data.shape[0], esn.n_out))
    outputs = np.zeros(targets.shape)
    for j in range(esn.test_data):
        instance = esn.test_data[j]
        x_in = np.vstack((np.zeros((1, esn.n_in)), instance[0]))
        targets[j] = instance[1]  # only a single vector for testing to match target classification for sentence
        x_instance = np.zeros((x_in.shape[0], esn.n_r))
        x_out = np.zeros((x_in.shape[0], esn.n_out))
        for i in range(1, x_in.shape[0] + 1):
            x_instance[i] = np.tanh((x_in[i] * esn.w_in)
                                    + (x_instance[i - 1] * esn.w_r)
                                    + (x_out[i - 1] * esn.w_fb))
            x_instance[i] = (1 - esn.alpha) * x_instance[i - 1] + esn.alpha * x_instance[i]
            x_out[i] = sigmoid(x_instance[i].dot(esn.w_out.T))

            # will maybe want a more efficient way of doing this:
            max_index = x_out[i].argmax()
            x_out[i] = np.zeros(x_out[i].shape)
            x_out[i][max_index] = 1

    # x_out is then the array of all output vectors over time. need to somehow determine
    # a single classification and store it at x_out[j]

    accuracy = compute_accuracy(targets, outputs)
    return targets, outputs, accuracy

def compute_accuracy(expected, actual):
    """
    Simple computation of accuracy taking the number of correct over total.

    Keyword arguments:
    expected    -- array of 1s and 0s, expected values
    actual      -- array of 1s and 0s, actual values, same size as expected
    """
    total = float(expected.shape[0])
    n_correct = np.sum(np.multiply(expected, actual))
    return n_correct / total

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    run_simulation()

class EchoStateNetwork:
    """
    An Echo State Network with an input layer, reservoir, and output layer with associated connections.

    Attributes:
        n_in            number of input units
        n_r             number of reservoir units
        n_out           number of readout units

        n_train         number of training instances
        n_test          number of testing instances

        w_in		    weights from input to reservoir                     : n_r   x n_in + 1
        w_r		        weights internal to reservoir                       : n_r   x n_r
        w_out		    weights from reservoir to readout                   : n_r   x n_out
        w_fb            weights from readout back into reservoir            : n_out x n_r

        test_size       percentage of input data that is the testing set
            NOTE: a train set will be constructed as complement of test set

        x_in_train      activations of input nodes over train time          : t_train x n_in
        x_in_test       activations of input nodes over test time           : t_test x n_in
        x_target_train  target activations of output nodes over train time  : t_train x n_out
        x_target_test   target activations of output nodes over test time   : t_test x n_out
        x_r             activations of reservoir nodes over time            : t_train x n_r
        x_out           activations of output nodes over time               : t_train x n_out
        x_target        target activations for training                     : t_train x n_out

        scale_in        scale factor of input weights
        scale_r		    scale factor of reservoir weights
        scale_fb        scale factor of feedback weights

        density_in      density coefficient for input-reservoir
        density_r       density coefficient for reservoir
        density_fb      density coefficient for out-reservoir

        alpha	        leaking rate of reservoir units
        rho		        desired spectral radius of reservoir

        seed 	        seed for RandomState
        rs              RandomState for random number generation
    """

    def __init__(self, data, seed=123, n_r=100, density_in=1.0, density_r=0.1,
                 density_fb=1.0, scale_in=1.0, scale_r=1.0, scale_fb=1.0, alpha=0.9,
                 rho=0.9, w_out=[], test_size=0.2, x_r=[], x_out=[], signal_reps=10):
        self.data = data
        self.n_in = self.data[0][0].shape[0]
        # add bias node:
        self.inputs = np.concatenate((np.ones((inputs.shape[0], 1)), inputs), axis=1)
        self.targets = targets
        self.n_out = self.targets.shape[1]
        self.seed = seed
        self.rs = RandomState(self.seed)
        self.n_r = n_r
        self.density_in = density_in
        self.density_r = density_r
        self.density_fb = density_fb
        self.scale_in = scale_in
        self.scale_r = scale_r
        self.scale_fb = scale_fb
        self.rho = rho
        self.alpha = alpha
        self.w_in = self.initialize_weights(self.n_in + 1, self.n_r, self.density_in, self.rs, self.scale_in)
        self.w_r = self.initialize_reservoir(self.n_r, self.density_r, self.rs, self.scale_r, self.rho)
        self.w_fb = self.initialize_weights(self.n_out, self.n_r, self.density_fb, self.rs, self.scale_fb)
        self.w_out = w_out
        self.test_size = test_size
        self.train_data, self.test_data = self.train_test_split(self.data, self.test_size)
        # create pseudo-signals of data:
        # self.x_in_train = self.create_pseudo_signal(self.x_in_train, signal_reps)
        # self.x_in_test = self.create_pseudo_signal(self.x_in_test, signal_reps)
        # self.x_target_train = self.create_pseudo_signal(self.x_target_train, signal_reps)
        # self.x_target_test = self.create_pseudo_signal(self.x_target_test, signal_reps)
        self.n_train = self.x_in_train.shape[0]
        self.n_test = self.x_in_test.shape[0]
        self.x_r = x_r
        self.x_out = x_out

    @staticmethod
    def initialize_weights(n_rows, n_cols, density, randomstate, scale):
        """
        Initialize a sparse random matrix of weights with dimensions
        n_rows x n_cols and specified density in range [-scale, scale].

        Keyword arguments:
        n_rows      -- number of rows
        n_cols      -- number of columns
        density     -- density of connections
        randomstate -- RandomState object for random number generation
        scale       -- absolute value of minimum/maximum weight value
        """
        weights = sparse.rand(n_rows, n_cols, density, random_state=randomstate)
        weights = 2 * scale * weights - scale * weights.ceil()
        return weights

    @staticmethod
    def initialize_reservoir(n_units, density, randomstate, scale, spec_rad):
        """
        Initialize a sparse random reservoir as a square matrix representing connections among
        the n_units neurons with connections having specified density in range [-scale, scale].

        The weights are generated until they achieve a spectral radius of at least 0.01;
        due to the iterative nature of scipy.sparse.linalg.eigs, values under this threshold
        are unstable and do not produce consistent results over time.

        Keyword arguments:
        n_units     -- number of reservoir nodes
        density     -- density of connections (default 0.1)
        randomstate -- RandomState object for random number generation (default RandomState(1))
        scale       -- absolute value of minimum/maximum weight value (default 1.0)
        spec_rad    -- desired spectral radius to scale to (default 1.0)
        """
        while True:
            weights = EchoStateNetwork.initialize_weights(n_units, n_units, density, randomstate, scale)
            if max(abs(eigs(weights)[0])) >= 0.01:
                break

        weights = EchoStateNetwork.scale_spectral_radius(weights, spec_rad)
        return weights

    @staticmethod
    def scale_spectral_radius(weights, spec_rad):
        """
        Scales the specified weight matrix to have the desired spectral radius.

        Keyword arguments:
        weights     -- weight array to scale
        spec_rad    -- desired spectral radius to scale to (default 1.0)
        """
        weights = spec_rad * (weights / max(abs(eigs(weights)[0])))
        return weights

    @staticmethod
    def create_pseudo_signal(data, reps):
        """
        Create a pseudo-signal from a data set by repeating each row the specified number of times.

        :param data: basis of pseudo-signal
        :param reps: number of times each data element is to be repeated
        :return: pseudo-signal of original data
        """
        signal = np.tile(data[0], (reps, 1))
        for i in range(1, data.shape[0]):
            signal = np.vstack((signal, np.tile(data[i], (reps, 1))))
        return signal

    @staticmethod
    def add_bias(data):
        biased = data
        for x in biased:
            x_inputs = x[0]
            np.concatenate((np.ones((x_inputs.shape[0], 1)), x_inputs), axis=1)
        return biased

    @staticmethod
    def train_test_split(data, test_size):
        random.shuffle(data)

        split_index = test_size * len(data)
        test_data = data[:split_index]
        train_data = data[split_index:]

        return train_data, test_data

"""
Need to:
    - formulate actual data set and use it
    - provide way of computing performance metrics
    - decide on sampling for performance metrics (sample only once at end of train?)
"""

"""
input sentence representation as repeated pulse trains?
"""
