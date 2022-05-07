import unittest
import numpy as np
from cnn_numpy.cnn_model.pooling import Pooling

class TestPooling(unittest.TestCase):
    def test_forward_pool(self):
        np.random.seed(1)
        A_prev = np.random.randn(2, 5, 5, 3)

        #test mean
        pool = Pooling(filter_shape=(3,3), mode='mean', stride=1)
        filter_shape = pool.hparams["filter_shape"]
        stride = pool.hparams["stride"]
        assert filter_shape == (3,3)
        assert stride == 1
        assert pool.mode == "mean"

        A_mean = pool.forward_pass(A_prev)
        exp_mean = np.array([[[[-3.01046719e-02, -3.24021315e-03, -3.36298859e-01],
                                         [1.43310483e-01, 1.93146751e-01, -
                                         4.44905196e-01],
                                         [1.28934436e-01, 2.22428468e-01, 1.25067597e-01]],

                                        [[-3.81801899e-01, 1.59993515e-02, 1.70562706e-01],
                                         [4.73707165e-02, 2.59244658e-02,
                                          9.20338402e-02],
                                         [3.97048605e-02, 1.57189094e-01, 3.45302489e-01]],

                                        [[-3.82680519e-01, 2.32579951e-01, 6.25997903e-01],
                                         [-2.47157416e-01, -3.48524998e-04,
                                          3.50539717e-01],
                                         [-9.52551510e-02, 2.68511000e-01, 4.66056368e-01]]],


                                       [[[-1.73134159e-01, 3.23771981e-01, -3.43175716e-01],
                                         [3.80634669e-02, 7.26706274e-02, -
                                         2.30268958e-01],
                                         [2.03009393e-02, 1.41414785e-01, -1.23158476e-02]],

                                        [[4.44976963e-01, -2.61694592e-03, -3.10403073e-01],
                                         [5.08114737e-01, -
                                         2.34937338e-01, -2.39611830e-01],
                                         [1.18726772e-01, 1.72552294e-01, -2.21121966e-01]],

                                        [[4.29449255e-01, 8.44699612e-02, -2.72909051e-01],
                                         [6.76351685e-01, -
                                         1.20138225e-01, -2.44076712e-01],
                                         [1.50774518e-01, 2.89111751e-01, 1.23238536e-03]]]])

        np.testing.assert_allclose(A_mean, exp_mean)

        #test max
        pool2 = Pooling(filter_shape=(3,3), mode='max', stride=1)
        A_max = pool2.forward_pass(A_prev)
        assert pool2.mode == "max"

        exp_max = np.array([[[[1.74481176, 0.90159072, 1.65980218],
                                     [1.74481176, 1.46210794, 1.65980218],
                                     [1.74481176, 1.6924546, 1.65980218]],

                                    [[1.14472371, 0.90159072, 2.10025514],
                                     [1.14472371, 0.90159072, 1.65980218],
                                     [1.14472371, 1.6924546, 1.65980218]],

                                    [[1.13162939, 1.51981682, 2.18557541],
                                     [1.13162939, 1.51981682, 2.18557541],
                                     [1.13162939, 1.6924546, 2.18557541]]],


                                   [[[1.19891788, 0.84616065, 0.82797464],
                                     [0.69803203, 0.84616065, 1.2245077],
                                     [0.69803203, 1.12141771, 1.2245077]],

                                    [[1.96710175, 0.84616065, 1.27375593],
                                     [1.96710175, 0.84616065, 1.23616403],
                                     [1.62765075, 1.12141771, 1.2245077]],

                                    [[1.96710175, 0.86888616, 1.27375593],
                                     [1.96710175, 0.86888616, 1.23616403],
                                     [1.62765075, 1.12141771, 0.79280687]]]])

        np.testing.assert_allclose(A_max, exp_max)

        #test cache
        cache = pool.cache["A"]
        np.testing.assert_allclose(cache, A_prev)

    def test_backward_pool(self):
        np.random.seed(1)
        #max pooling
        pool = Pooling(filter_shape=(2,2),mode='max',stride=1)
        A_prev = np.random.randn(5, 5, 3, 2)
        #hparameters = {"stride" : 1, "f": 2}
        A = pool.forward_pass(A_prev)
        print(A.shape)
        dA = np.random.randn(5, 4, 2, 2)

        dA_prev1 = pool.backward_pass(dA)
        print("mode = max")
        print('mean of dA = ', np.mean(dA))
        print('dA_prev1[1,1] = ', dA_prev1[1, 1])
        print("------------------")

        #average pooling
        pool2 = Pooling(filter_shape=(2,2),mode='mean',stride=1)
        A2 = pool2.forward_pass(A_prev)
        print(A2)
        dA_prev2 = pool2.backward_pass(dA)
        print("mode = average")
        print('mean of dA = ', np.mean(dA))
        print('dA_prev2[1,1] = ', dA_prev2[1, 1])

        assert type(dA_prev1) == np.ndarray, "Wrong type"
        assert dA_prev1.shape == (5, 5, 3, 2), f"Wrong shape {dA_prev1.shape} != (5, 5, 3, 2)"
        assert np.allclose(dA_prev1[1, 1], [[0, 0],
                                            [ 5.05844394, -1.68282702],
                                            [ 0, 0]]), "Wrong values for mode max"
        assert np.allclose(dA_prev2[1, 1], [[0.08485462,  0.2787552],
                                            [1.26461098, -0.25749373],
                                            [1.17975636, -0.53624893]]), "Wrong values for mode average"



if __name__ == '__main__':
    unittest.main()
