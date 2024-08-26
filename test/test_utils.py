import unittest
import numpy as np
import os, sys

sys.path.append(os.path.abspath("../src/"))

from utils import valid_values

class TestValidValues(unittest.TestCase):
    def test_no_nan_or_inf(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertTrue(valid_values(arr))

    def test_contains_nan(self):
        arr = np.array([1.0, 2.0, np.nan, 4.0])
        self.assertFalse(valid_values(arr))

    def test_contains_inf(self):
        arr = np.array([1.0, 2.0, np.inf, 4.0])
        self.assertFalse(valid_values(arr))

    def test_contains_negative_inf(self):
        arr = np.array([1.0, 2.0, -np.inf, 4.0])
        self.assertFalse(valid_values(arr))

    def test_empty_array(self):
        arr = np.array([])
        self.assertTrue(valid_values(arr))

    def test_2d_array_no_nan_or_inf(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.assertTrue(valid_values(arr))

    def test_2d_array_contains_nan(self):
        arr = np.array([[1.0, 2.0], [np.nan, 4.0]])
        self.assertFalse(valid_values(arr))

    def test_2d_array_contains_inf(self):
        arr = np.array([[1.0, 2.0], [np.inf, 4.0]])
        self.assertFalse(valid_values(arr))

    def test_3d_array_no_nan_or_inf(self):
        arr = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        self.assertTrue(valid_values(arr))

    def test_3d_array_contains_nan(self):
        arr = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, np.nan], [7.0, 8.0]]])
        self.assertFalse(valid_values(arr))

    def test_3d_array_contains_inf(self):
        arr = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, np.inf], [7.0, 8.0]]])
        self.assertFalse(valid_values(arr))

if __name__ == '__main__':
    unittest.main()