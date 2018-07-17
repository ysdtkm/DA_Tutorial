#!/usr/bin/env python3

import pickle
import unittest
import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

DIR_RESULTS = "/lustre/tyoshida/six_month/result_exempt"

def read_error_ETKF_or_hybrid(parent_dir, expname):
    with open(f"{parent_dir}/{expname}/plot_reduced_rmse.pkl", "rb") as f:
        obj = pickle.load(f)
    return obj

def read_error_3DVar(parent_dir, expname):
    obj = np.load(f"{parent_dir}/{expname}/rmse_3DVar.npy")
    return obj

class TestAll(unittest.TestCase):
    def test_read_error_ETKF_or_hybrid(self):
        for exp in ["t0900", "t0901", "t0902", "t0931", "t0932", "t0933"]:
            res = read_error_ETKF_or_hybrid(DIR_RESULTS, exp)
            self.assertTrue(isinstance(res, list))

    def test_read_error_3DVar(self):
        for exp in ["t0942", "t0944", "t0945"]:
            res = read_error_3DVar(DIR_RESULTS, exp)
            self.assertTrue(isinstance(res, np.ndarray))

if __name__ == "__main__":
    unittest.main()
