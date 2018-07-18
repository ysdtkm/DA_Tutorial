#!/usr/bin/env python3

import unittest
import numpy as np
from module_constants import get_x_std, get_static_b, read_text_b
from exp_params import SEED, SIGMA_B

class TestStaticB(unittest.TestCase):
    def test_static_b(self):
        for i in range(10):
            np.random.seed(SEED * 3)
            B1 = get_static_b()
            B2 = get_static_b()
            di = np.max(np.abs(B1 - B2))
            self.assertEqual(di, 0.0, i)

    def test_eigvals(self):
        b0 = read_text_b("binary_const/20180702_cheng_b.txt")
        for i in range(100):
            e1 = np.linalg.eigvals(b0)
            e2 = np.linalg.eigvals(b0)
            self.assertEqual(np.max(np.abs(e1 - e2)), 0.0, i)

            m1 = np.max(e1)
            m2 = np.max(e2)
            self.assertEqual(m1 - m2, 0.0, i)

            b1 = b0 / np.max(e1)
            b2 = b0 / np.max(e2)
            self.assertEqual(np.max(np.abs(b1 - b2)), 0.0, i)

    def test_read_text_b(self):
        for i in range(10):
            np.random.seed(SEED * 3)
            B1 = read_text_b("binary_const/20180702_cheng_b.txt")
            B2 = read_text_b("binary_const/20180702_cheng_b.txt")
            di = np.max(np.abs(B1 - B2))
            self.assertEqual(di, 0.0, i)

