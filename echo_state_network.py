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

w_in		    weights from input to reservoir             : n_in + 1 x n_r
w_r		        weights internal to reservoir               : n_r      x n_r
w_out		    weights from reservoir to readout           : n_r      x n_out
w_fb            weights from readout back into reservoir    : n_out    x n_r

x_in            activations of input nodes over time        : n_in  x t_train
x_r             activations of reservoir nodes over time    : n_r   x t_train
x_out           activations of output nodes over time       : n_out x t_train

scale_in        scale factor of input weights
scale_r		    scale factor of reservoir weights
scale_fb        scale factor of feedback weights

alpha	        leaking rate of reservoir units
rho		        desired spectral radius of reservoir
density 	    sparsity coefficient for reservoir

seed 	        seed for RandomState
rs              RandomState for random number generation
==========================================================
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs

data = np.array()               # THIS IS ALL TEMPORARY, will need module to actually import data
m_train, n_in = data.shape      # and set everything appropriately
t_train = 1000                  #
x_in = np.array(n_in, t_train)  #
# add bias node:
x_in = np.concatenate((x_in, np.ones((1, t_train))))

n_r = 100
n_out = 1

scale_in = 1
scale_r = 1
scale_fb = 1

density_in = 1.0     # experiment with input sparsity, may get better speed without negative effect
density_r = 0.1
density_fb = 1.0     # experiment with feedback sparsity

alpha = 1
rho = 1         # NEED TO check what are good default values

seed = 123
rs = np.random.RandomState(seed)

# initialize input weights:
w_in = np.random.rand(n_in + 1, n_r, density_in, random_state=rs)
# put into range [-scale_in, scale_in]
w_in = 2 * scale_in * w_in - scale_in * w_in.ceil()

# initialize sparse random reservoir:
w_r = sparse.rand(n_r, n_r, density_r, random_state=rs)
# put into range [-scale_r, scale_r]:
w_r = 2 * scale_r * w_r - scale_r * w_r.ceil()
# scale to have desired spectral radius:
w_r = rho * (w_r / max(abs(eigs(w_r)[0])))

# initialize feedback weights:
w_fb = sparse.rand(n_out, n_r, density_fb, random_state=rs)
# put into range [-scale_fb, scale_fb]
w_fb = 2 * scale_fb * w_fb - scale_fb * w_fb.ceil()

# initialize reservoir and output nodes to 0:
x_r = np.zeros(n_r, 1)
x_out = np.zeros(n_out, 1)

# compute reservoir states over train duration:
for i in range(1, t_train):
    x_r[:, i] = np.tanh(w_in.T.dot(x_in[:, i])
                        + w_r.dot(x_r[:, i-1])
                        + w_fb.dot(x_out[:, i-1]))


"""
STATUS: Need to collect reservoir states and do ridge regression
"""

"""
input sentence representation as repeated pulse trains?
"""