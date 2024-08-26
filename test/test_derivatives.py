import numpy as np
import os
import sys
import unittest
import logging

sys.path.append(os.path.abspath("../src/"))

from derivatives import hessian, jacobian, TransformType

class TestJacobian(unittest.TestCase):
    
    def test_translation(self):
        nx = 2
        ny = 2
        transform_type = TransformType.TRANSLATION
        J = jacobian(transform_type, nx, ny)
        self.assertEqual(J.shape, (ny, nx, 2 * transform_type.nparams()))
        np.testing.assert_array_almost_equal(J,
            np.array([
                [[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]]
                ])
            )

    def test_euclidean(self):
        nx = 2
        ny = 2
        transform_type = TransformType.EUCLIDEAN
        J = jacobian(transform_type, nx, ny)
        self.assertEqual(J.shape, (ny, nx, 2 * transform_type.nparams()))
        np.testing.assert_array_almost_equal(
            J,
            np.array([
                [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 1.0]],
                [[1.0, 0.0, -1.0, 0.0, 1.0, 0.0], [1.0, 0.0, -1.0, 0.0, 1.0, 1.0]]
                ])
            )

    def test_similarity(self):
        nx = 2
        ny = 2
        transform_type = TransformType.SIMILARITY
        J = jacobian(transform_type, nx, ny)
        self.assertEqual(J.shape, (ny, nx, 2 * transform_type.nparams()))
        np.testing.assert_array_almost_equal(
            J,
            np.array([
                [[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0, -1.0, 0.0, 1.0, 1.0, 0.0], [1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 1.0]]
                ])
            )

    def test_affinity(self):
        nx = 2
        ny = 2
        transform_type = TransformType.AFFINITY
        J = jacobian(transform_type, nx, ny)
        self.assertEqual(J.shape, (ny, nx, 2 * transform_type.nparams()))
        np.testing.assert_array_almost_equal(
            J,
            np.array([
                    [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]],
                    [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]]
                ])
            )


class TestHessianFunction(unittest.TestCase):
    
    def test_valid_dij(self):
        DIJ = np.random.rand(3, 3, 4, 2)
        expected_H = np.zeros((2, 2), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                DIJ_slice = DIJ[i, j, :, :]
                if not np.isnan(DIJ_slice).any() and not np.isinf(DIJ_slice).any():
                    expected_H += DIJ_slice.T @ DIJ_slice
        result = hessian(DIJ)
        np.testing.assert_array_almost_equal(result, expected_H)
    
    def test_invalid_dij(self):
        DIJ = np.random.rand(3, 3, 4, 2)
        DIJ[0, 0, 0, 0] = np.nan  # Introduce a NaN value
        DIJ[1, 1, 1, 1] = np.inf  # Introduce an infinite value
        expected_H = np.zeros((2, 2), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                DIJ_slice = DIJ[i, j, :, :]
                if not np.isnan(DIJ_slice).any() and not np.isinf(DIJ_slice).any():
                    expected_H += DIJ_slice.T @ DIJ_slice
        result = hessian(DIJ)
        np.testing.assert_array_almost_equal(result, expected_H)
    
    def test_empty_dij(self):
        DIJ = np.zeros((0, 0, 0, 0))
        expected_H = np.zeros((0, 0), dtype=np.float64)
        result = hessian(DIJ)
        np.testing.assert_array_almost_equal(result, expected_H)


def main():
    unittest.main()

if __name__ == '__main__':
    main()