#!/usr/bin/env python3

import pickle
import unittest
import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

DIR_RESULTS = "/lustre/tyoshida/six_month/result_exempt"
OBS_LIST = ["full", "atmos", "ocean"]
DA_LIST = ["hybrid", "ETKF", "3DVar"]
EDIM_LIST = list(range(2, 38))

def get_exp_dict(obs, da):
    exp_dict = {
        ("full", "hybrid"): "t0902",
        ("atmos", "hybrid"): "t0901",
        ("ocean", "hybrid"): "t0900",
        ("full", "ETKF"): "t0933",
        ("atmos", "ETKF"): "t0931",
        ("ocean", "ETKF"): "t0932",
        ("full", "3DVar"): "t0942",
        ("atmos", "3DVar"): "t0944",
        ("ocean", "3DVar"): "t0945",
    }
    res = exp_dict[(obs, da)]
    return res

def read_error_ETKF_or_hybrid(parent_dir, expname):
    with open(f"{parent_dir}/{expname}/plot_reduced_rmse.pkl", "rb") as f:
        obj = pickle.load(f)
    return obj

def read_error_3DVar(parent_dir, expname):
    obj = np.load(f"{parent_dir}/{expname}/rmse_3DVar.npy")
    return obj

def get_rmse_comparison(obs):
    assert obs in OBS_LIST
    index_rmse_component = 4
    rmse_3dvar = read_error_3DVar(DIR_RESULTS, get_exp_dict(obs, "3DVar"))[index_rmse_component]

    rmse_ETKF_hybrid = {}
    for da in ["ETKF", "hybrid"]:
        rmse_2dlist = read_error_ETKF_or_hybrid(DIR_RESULTS, get_exp_dict(obs, da))[-1]
        assert len(rmse_2dlist) == len(EDIM_LIST)
        rmse_ndarray = np.empty(len(EDIM_LIST))
        for i in range(len(EDIM_LIST)):
            rmse_ndarray[i] = rmse_2dlist[i][0][index_rmse_component]
        rmse_ETKF_hybrid[da] = rmse_ndarray
    return rmse_3dvar, rmse_ETKF_hybrid

def plot_rmse_comparison(obs):
    rmse_3dvar, rmse_ETKF_hybrid = get_rmse_comparison(obs)
    print(rmse_ETKF_hybrid)

class TestAll(unittest.TestCase):
    def test_read_error_ETKF_or_hybrid(self):
        for obs in OBS_LIST:
            for da in ["hybrid", "ETKF"]:
                exp = get_exp_dict(obs, da)
                res = read_error_ETKF_or_hybrid(DIR_RESULTS, exp)
                self.assertTrue(isinstance(res, list))

    def test_read_error_3DVar(self):
        for obs in OBS_LIST:
            exp = get_exp_dict(obs, "3DVar")
            res = read_error_3DVar(DIR_RESULTS, exp)
            self.assertTrue(isinstance(res, np.ndarray))

    def test_get_exp_dict(self):
        for obs in OBS_LIST:
            for da in DA_LIST:
                res = get_exp_dict(obs, da)
                self.assertTrue(isinstance(res, str))

if __name__ == "__main__":
    plot_rmse_comparison("full")
