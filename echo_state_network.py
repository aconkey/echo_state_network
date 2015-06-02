# -*- coding: utf-8 -*-
"""
Created on Sat May  2 21:07:54 2015

@author: adam

==================== PARAMETERS ====================

n_in            number of input units
n_r             number of reservoir units
n_out           number of readout units

w_in		    weights from input to reservoir
w_r		        weights internal to reservoir
w_out		    weights from reservoir to readout

scale_in        scale factor of input weights
scale_r		    scale factor of reservoir weights

alpha	        leaking rate of reservoir units
rho		        desired spectral radius of reservoir
density 	    sparsity coefficient for reservoir

seed 	        seed for RandomState
rs              RandomState for random number generation
====================================================
"""
from numpy.random import RandomState
from scipy import sparse

n_in = 10
n_r = 100
n_out = 1

density = 0.1

seed = 123
rs = RandomState(seed)

# initialize sparse random reservoir:
w_r = sparse.rand(n_r, n_r, density, random_state=rs)
