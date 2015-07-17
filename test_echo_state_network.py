import unittest
import echo_state_network as esn
from echo_state_network import EchoStateNetwork
from numpy.random import RandomState
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal
from scipy.sparse.linalg import eigs
import numpy as np

class EchoStateNetworkTests(unittest.TestCase):

    def setUp(self):
        self.rs1 = RandomState(1)
        self.rs2 = RandomState(1)

    def test_initialize_weights(self):
        w1 = EchoStateNetwork.initialize_weights(10, 10, 0.1, self.rs1, 1.0)
        w2 = EchoStateNetwork.initialize_weights(10, 10, 0.1, self.rs2, 1.0)
        assert_allclose(w1.toarray(), w2.toarray())
        self.assertLessEqual(np.amax(w1.data), 1.0)
        self.assertGreaterEqual(np.amin(w1.data), -1.0)

        w3 = EchoStateNetwork.initialize_weights(10, 10, 0.1, self.rs1, 5.0)
        self.assertLessEqual(np.amax(w3.data), 5.0)
        self.assertGreaterEqual(np.amin(w3.data), -5.0)
        self.assertLess(np.amin(w3.data), -1.0)
        self.assertGreater(np.amax(w3.data), 1.0)

    def test_initialize_reservoir(self):
        w1 = EchoStateNetwork.initialize_reservoir(10, 0.1, self.rs1, 1.0, 1.0)
        self.assertAlmostEqual(1.0, max(abs(eigs(w1)[0])))
        w2 = EchoStateNetwork.initialize_reservoir(10, 0.1, self.rs1, 1.0, 5.0)
        self.assertAlmostEqual(5.0, max(abs(eigs(w2)[0])))
        w3 = EchoStateNetwork.initialize_reservoir(10, 0.1, self.rs1, 1.0, 0.1)
        self.assertAlmostEqual(0.1, max(abs(eigs(w3)[0])))

    def test_compute_accuracy(self):
        expected = np.array([[1, 0, 0],
                             [0, 1, 0],
                             [1, 0, 0],
                             [0, 0, 1],
                             [0, 0, 1],
                             [0, 1, 0],
                             [1, 0, 0],
                             [1, 0, 0]])
        a1 = expected
        a2 = np.array([[0, 1, 0],
                       [0, 1, 0],
                       [1, 0, 0],
                       [0, 0, 1],
                       [0, 0, 1],
                       [0, 1, 0],
                       [1, 0, 0],
                       [0, 1, 0]])
        a3 = np.array([[0, 1, 0],
                       [1, 0, 0],
                       [0, 1, 0],
                       [1, 0, 0],
                       [0, 1, 0],
                       [1, 0, 0],
                       [0, 1, 0],
                       [0, 1, 0]])
        a4= np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 1, 0],
                      [0, 1, 0],
                      [0, 1, 0],
                      [1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])

        self.assertAlmostEqual(1.0, esn.compute_accuracy(expected, a1))
        self.assertAlmostEqual(0.75, esn.compute_accuracy(expected, a2))
        self.assertAlmostEqual(0.0, esn.compute_accuracy(expected, a3))
        self.assertAlmostEqual(0.25, esn.compute_accuracy(expected, a4))

    def test_basic_run_simulation(self):
        inputs = np.random.rand(10, 10)
        targets = np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [0, 1]])
        network = esn.EchoStateNetwork(inputs, targets)
        print '\nRunning basic simulation...'
        accuracy = esn.run_simulation(network)
        print 'Test accuracy:'
        print accuracy

    # def test_load_run_simulation(self):
    #     inputs = np.random.rand(100000, 300)
    #     targets = np.ones((100000, 2))
    #     network = esn.EchoStateNetwork(inputs, targets)
    #     print '\nRunning stress load simulation...'
    #     accuracy = esn.run_simulation(network)
    #     print 'Test accuracy:'
    #     print accuracy

    def test_create_pseudo_signal(self):
        orig = np.array([[0, 1, 0],
                         [1, 0, 0]])
        repeat = np.array([[0, 1, 0],
                           [0, 1, 0],
                           [0, 1, 0],
                           [1, 0, 0],
                           [1, 0, 0],
                           [1, 0, 0]])
        print 'Original:'
        print orig
        print 'Repeated:'
        print repeat

        assert_array_equal(repeat, esn.create_pseudo_signal(orig, 3))

def main():
    unittest.main()

if __name__ == '__main__':
    main()