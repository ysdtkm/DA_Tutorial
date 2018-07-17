#!/usr/bin/env python3

import pickle
import unittest
import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

def read_error_ETKF_or_hybrid(parent_dir, expname):
    with open(f"{parent_dir}/{expname}/plot_reduced_rmse.pkl", "rb") as f:
        obj = pickle.load(f)
    return obj

class TestAll(unittest.TestCase):
    def test_read_error_ETKF_or_hybrid(self):
        parent_dir = "/lustre/tyoshida/six_month/result_exempt"
        exp = "t0900"
        print(read_error_ETKF_or_hybrid(parent_dir, exp))

"""[tyoshida@login-1:/lustre/tyoshida/six_month/result_exempt ]
$ find .
.
./t0902
./t0902/plot_reduced_rmse.pkl
./t0945
./t0945/rmse_3DVar.npy
./t0900
./t0900/plot_reduced_rmse.pkl
./t0932
./t0932/plot_reduced_rmse.pkl
./t0901
./t0901/plot_reduced_rmse.pkl
./t0942
./t0942/rmse_3DVar.npy
./t0944
./t0944/rmse_3DVar.npy
./t0931
./t0931/plot_reduced_rmse.pkl
./t0933
./t0933/plot_reduced_rmse.pkl"""

if __name__ == "__main__":
    unittest.main()
