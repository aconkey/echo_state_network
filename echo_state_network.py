# -*- coding: utf-8 -*-
"""
Created on Sat May  2 21:07:54 2015

@author: Adam Conkey

======================= PARAMETERS =======================

n_in            number of input units
n_r             number of reservoir units
n_out           number of readout units

w_in		    weights from input to reservoir
w_r		        weights internal to reservoir
w_out		    weights from reservoir to readout
w_fb            weights from readout back into reservoir

scale_in        scale factor of input weights
scale_r		    scale factor of reservoir weights

alpha	        leaking rate of reservoir units
rho		        desired spectral radius of reservoir
density 	    sparsity coefficient for reservoir

seed 	        seed for RandomState
rs              RandomState for random number generation
==========================================================
"""
from numpy.random import RandomState
from scipy import sparse
from scipy.sparse.linalg import eigs

n_in = 10
n_r = 100
n_out = 1

alpha = 1
rho = 1         # NEED TO check what are good default values
density = 0.1

seed = 123
rs = RandomState(seed)

# initialize sparse random reservoir:
w_r = sparse.rand(n_r, n_r, density, random_state=rs)
# put into range [-1,1]:
w_r = 2 * w_r - w_r.ceil()
# scale to have desired spectral radius:
w_r = rho * (w_r / max(abs(eigs(w_r)[0])))


"""
STATUS: Reservoir is good to go. Need to connect inputs and readouts.
"""

"""
input sentence representation as repeated pulse trains?
"""