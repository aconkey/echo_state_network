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

    def test_scale_spectral_radius(self):
        w1 = esn.initialize_weights(10, 10)
        w1 = esn.scale_spectral_radius(w1)
        self.assertAlmostEqual(1.0, max(abs(eigs(w1)[0])))
        w2 = esn.initialize_weights(10, 10)
        w2 = esn.scale_spectral_radius(w2, 5.0)
        self.assertAlmostEqual(5.0, max(abs(eigs(w2)[0])))
        w3 = esn.initialize_weights(10, 10)
        w3 = esn.scale_spectral_radius(w3, 0.1)
        self.assertAlmostEqual(0.1, max(abs(eigs(w3)[0])))

def main():
    unittest.main()

if __name__ == '__main__':
    main()