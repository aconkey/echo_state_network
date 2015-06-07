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
from sklearn.linear_model import Ridge

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
x_r = np.zeros(1, n_r)
x_out = np.zeros(1, n_out)

# compute reservoir states over train duration:
for i in range(1, t_train):
    x_r[i] = np.tanh(x_in[i].dot(w_in)
                     + x_r[i-1].dot(w_r)
                     + x_target[i-1].dot(w_fb))
    x_r[i] = (1 - alpha) * x_r[i-1] + alpha * x_r[i]


# compute the output weights using Ridge Regression:
clf = Ridge()                   # play with parameters ???
clf.fit(x_r, x_target)
w_out = clf.coef_

"""
STATUS: Ridge Regression done.

Need to:
    - drive with novel input
    - break into actual modules
    - create mock data set for testing and import it
    - formulate actual data set and use it
    - provide way of computing performance metrics
    - decide on sampling for performance metrics (sample only once at end of train?)
"""

"""
input sentence representation as repeated pulse trains?
"""