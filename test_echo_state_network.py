import unittest
import echo_state_network as esn
from numpy.random import RandomState
from numpy.testing import assert_allclose
from scipy.sparse.linalg import eigs
import numpy as np

class EchoStateNetworkTests(unittest.TestCase):

    def setUp(self):
        self.rs1 = RandomState(1)
        self.rs2 = RandomState(1)

    def test_initialize_weights(self):
        w1 = esn.initialize_weights(10, 10, randomstate=self.rs1)
        w2 = esn.initialize_weights(10, 10, randomstate=self.rs2)
        assert_allclose(w1.toarray(), w2.toarray())
        self.assertLessEqual(np.amax(w1.data), 1.0)
        self.assertGreaterEqual(np.amin(w1.data), -1.0)

        w3 = esn.initialize_weights(10, 10, randomstate=self.rs1, scale=5.0)
        self.assertLessEqual(np.amax(w3.data), 5.0)
        self.assertGreaterEqual(np.amin(w3.data), -5.0)
        self.assertLess(np.amin(w3.data), -1.0)
        self.assertGreater(np.amax(w3.data), 1.0)

    def test_initialize_reservoir(self):
        w1 = esn.initialize_reservoir(10)
        self.assertAlmostEqual(1.0, max(abs(eigs(w1)[0])))
        w2 = esn.initialize_reservoir(10, spec_rad=5.0)
        self.assertAlmostEqual(5.0, max(abs(eigs(w2)[0])))
        w3 = esn.initialize_reservoir(10, spec_rad=0.1)
        self.assertAlmostEqual(0.1, max(abs(eigs(w3)[0])))

    def test_compute_accuracy(self):
        e1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        e2 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        e3 = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
        e4 = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
        e5 = np.array([1, 1, 0, 1, 0, 0, 1, 0, 0, 1])
        a1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        a2 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        a3 = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
        a4 = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
        a5 = np.array([1, 0, 0, 1, 0, 1, 1, 1, 0, 0])
        self.assertAlmostEqual(1.0, esn.compute_accuracy(e1, a1))
        self.assertAlmostEqual(1.0, esn.compute_accuracy(e2, a2))
        self.assertAlmostEqual(1.0, esn.compute_accuracy(e3, a3))
        self.assertAlmostEqual(1.0, esn.compute_accuracy(e4, a4))
        self.assertAlmostEqual(0.7, esn.compute_accuracy(e3, a2))
        self.assertAlmostEqual(0.7, esn.compute_accuracy(e4, a1))
        self.assertAlmostEqual(0.0, esn.compute_accuracy(e1, a2))
        self.assertAlmostEqual(0.0, esn.compute_accuracy(e2, a1))
        self.assertAlmostEqual(0.0, esn.compute_accuracy(e3, a4))
        self.assertAlmostEqual(0.0, esn.compute_accuracy(e4, a3))
        self.assertAlmostEqual(0.6, esn.compute_accuracy(e5, a5))

    def test_run_simulation(self):
        inputs = np.random.rand(10, 10)
        targets = np.array([[1],
                            [1],
                            [1],
                            [0],
                            [0],
                            [0],
                            [1],
                            [1],
                            [1],
                            [1]])
        data = (inputs, targets)
        network = esn.EchoStateNetwork()
        print 'Running simulation:'
        esn.run_simulation(network, data)

def main():
    unittest.main()

if __name__ == '__main__':
    main()