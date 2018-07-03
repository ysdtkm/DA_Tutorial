#!/usr/bin/env python3

import sys
import numpy as np

NDIM = 36

def get_x_std():
    x_std = np.array(
        [2.64866459e-04, 2.08002612e-02, 2.91309882e-02, 2.60987125e-02,
         2.70879647e-03, 1.10145131e-02, 5.11252252e-03, 6.90632988e-03,
         3.80084735e-03, 1.89289705e-03, 1.29020344e-03, 9.11129276e-03,
         1.39798460e-02, 1.33605394e-02, 2.66163757e-03, 5.68211203e-03,
         1.78791818e-03, 3.33561478e-03, 8.28188515e-04, 7.71253940e-04,
         3.33392496e-08, 5.18179764e-09, 4.37903364e-08, 2.40714819e-10,
         5.00145159e-08, 4.81824999e-08, 3.01073478e-09, 1.03304367e-10,
         4.12804966e-04, 6.92212257e-05, 3.46594115e-04, 8.34432052e-06,
         6.98347756e-04, 1.35828889e-04, 1.21582441e-07, 2.24123574e-08])
    return x_std

def get_static_b():
    # bcov = np.load("binary_const/mean_b_cov_t0644_0001tu.npy")
    # bcov = np.load("binary_const/mean_b_cov_t0645_0010tu.npy")
    # bcov = np.load("binary_const/mean_b_cov_t0646_0100tu.npy")
    # bcov = np.load("binary_const/mean_b_cov_t0647_1000tu.npy")
    # bcov = np.load("binary_const/mean_b_cov_t0800_6h.npy")
    bcov = read_text_b("binary_const/20180702_cheng_b.txt")
    # bcov = np.load("binary_const/mean_b_cov_t0829_6h_ocnobs.npy")
    # bcov = np.load("binary_const/mean_b_cov_t0830_6h_atmobs.npy")
    # bcov = np.identity(NDIM)
    assert bcov.shape == (NDIM, NDIM)
    eigs = np.linalg.eigvals(bcov)
    assert np.all(eigs > 0.0)
    srad = np.max(eigs)
    bcov /= srad
    return bcov

def read_text_b(filename):
    b = np.zeros((NDIM, NDIM))
    with open(filename, "r") as f:
        for i in range(NDIM):
            ls = f.readline().replace("D", "E").split()
            lf = list(map(float, ls))
            assert len(lf) == NDIM
            b[i, :] = np.array(lf)
    return b

if __name__ == "__main__":
    get_static_b()
