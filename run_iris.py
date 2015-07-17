import echo_state_network as esn
import numpy as np
from sklearn import preprocessing

inputs = np.genfromtxt('resources/iris_inputs.csv', delimiter=",")
targets = np.genfromtxt('resources/iris_targets.csv', delimiter=",")
inputs = preprocessing.scale(inputs)
network = esn.EchoStateNetwork(inputs, targets, n_r=10000, density_r=0.1, scale_in=1.0,
                               scale_r=1.0, scale_fb=1.0, alpha=0.9, rho=1.0, signal_reps=1)
print '\nRunning Iris data set...'
accuracy = esn.run_simulation(network)
print 'Test accuracy:'
print accuracy
