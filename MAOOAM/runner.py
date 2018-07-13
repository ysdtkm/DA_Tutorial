#!/usr/bin/env python3

import collections
import functools
import itertools
from multiprocessing import Pool, cpu_count
import os
import sys
import subprocess as sp
import numpy as np

INPATH = "/lustre/tyoshida/prgm/DA_Tutorial/MAOOAM"
WKPATH = "/lustre/tyoshida/shrt/MAOOAM_WK"

def main_parallel():
    params = [
        ParameterAxis("ensemble member", list(range(2, 38, 12)), "%02d", [Rewriter("exp_params.py", 16, "EDIM", "EDIM = {param:d}")]),
    ]
    res = Runner.run_parallel(params, "make", 4)
    visualize_1d_rmse(params, res)

# def plot_reduced_rmse(params1, params2, name1, name2, res):
#     if params1 is params2 is res is None:
#         with open("plot_reduced_rmse.pkl", "rb") as f:
#             params1, params2, name1, name2, res = pickle.load(f)
#     else:
#         with open("plot_reduced_rmse.pkl", "wb") as f:
#             pickle.dump([params1, params2, name1, name2, res], f)
#     n1, n2 = len(params1), len(params2)
#     nr = len(res[0][0])
#     res_npy = np.empty((nr, n1, n2))
#     names = ["atmos_psi", "atmos_temp", "ocean_psi", "ocean_temp", "all"]
#     for i1 in range(n1):
#         for i2 in range(n2):
#             res_npy[:, i1, i2] = res[i1][i2]
#     for ir in range(nr):
#         fig, ax = plt.subplots()
#         cm = plt.imshow(res_npy[ir, :, :], cmap="Reds", norm=matplotlib.colors.LogNorm())
#         # cm.set_clim(0, 0.2)
#         plt.colorbar(cm)
#         plt.xlabel(name2)
#         ax.set_xticks(range(n2))
#         ax.set_xticklabels(params2, rotation=90)
#         plt.ylabel(name1)
#         ax.set_yticks(range(n1))
#         ax.set_yticklabels(params1)
#         plt.savefig("out/rmse_%s.pdf" % names[ir], bbox_inches="tight")
#         plt.close()
# 
#         if n2 >= 2:
#             continue
#         plt.loglog(params1, res_npy[ir, :, 0])
#         plt.xlabel(name1)
#         # plt.ylim([0, 0.01])
#         plt.ylabel("RMS error")
#         plt.savefig("out/rmse_onedim_%s.pdf" % names[ir], bbox_inches="tight")
#         plt.close()

def visualize_1d_rmse(params, res):
    assert len(params) == 1
    assert isinstance(res, np.ndarray)
    import pdb; pdb.set_trace()
    subtract_rmse_entire_system = np.vectorize(lambda x: x[0])
    print(subtract_rmse_entire_system(res))

def get_output_obj():
    npa = np.load("rmse_ETKF.npy")
    return npa

def get_failed_obj():
    return "Failed"

# Following are backend. Generally no need to edit.
class Rewriter:
    def __init__(self, relative_file_name, line_num, match, formattable_replacer):
        assert isinstance(relative_file_name, str)
        assert isinstance(line_num, int)
        assert isinstance(match, str)
        assert isinstance(formattable_replacer, str)
        self.relative_file_name = relative_file_name
        self.line_num = line_num
        self.match = match
        self.formattable_replacer = formattable_replacer

    def rewrite_file_with_param(self, param):
        with open(self.relative_file_name, "r") as f:
            lines = f.readlines()
        with open(self.relative_file_name, "w") as f:
            for i, l in enumerate(lines):
                if i == self.line_num - 1:
                    assert self.match in l
                    f.write(self.formattable_replacer.format(param=param) + "\n")
                else:
                    f.write(l)

class ParameterAxis:
    def __init__(self, name, values, path_fmt, rewriters):
        assert isinstance(values, collections.Iterable)
        assert isinstance(rewriters, collections.Iterable)
        self.name = name
        self.path_fmt = path_fmt
        self.values = values
        self.rewriters = rewriters

class Runner:
    @classmethod
    def inverse_itertools_kd_product(cls, param_vals, product_objs):
        ns = [len(pv) for pv in param_vals]
        assert len(product_objs) == np.prod(ns, dtype=int)
        indices = [list(range(n)) for n in ns]
        product_indices = list(itertools.product(*indices))
        kd_array = np.empty(ns, dtype=object)
        for i, res in enumerate(product_objs):
            idx = product_indices[i]
            kd_array[idx] = product_objs[i]
        return kd_array

    @classmethod
    def shell(cls, cmd, writeout=False):
        assert isinstance(cmd, str)
        p = sp.run(cmd, shell=True, encoding="utf8", stderr=sp.PIPE, stdout=sp.PIPE)
        if writeout:
            with open("stdout", "w") as f:
                f.write(p.stdout)
            with open("stderr", "w") as f:
                f.write(p.stdout)
        else:
            if p.returncode != 0:
                print(p.stdout)
                print(p.stderr)
        if p.returncode != 0:
            raise Exception("shell %s failed" % cmd)

    @classmethod
    def exec_single_job(cls, params, command, list_param_val):
        k_dim = len(params)
        assert len(list_param_val) == k_dim
        str_path_part = [(params[j].path_fmt % list_param_val[j]).replace(".", "_") for j in range(k_dim)]
        suffix_path = "".join([f"/{str_path_part[j]}" for j in range(k_dim)])
        single_dir_name = f"{WKPATH}{suffix_path}"
        cls.shell(f"mkdir -p {single_dir_name}")
        os.chdir(single_dir_name)
        cls.shell(f"cp -rf {INPATH}/* .")
        for j in range(k_dim):
            for r in params[j].rewriters:
                r.rewrite_file_with_param(list_param_val[j])
        try:
            cls.shell(command, writeout=True)
            res = get_output_obj()
            print(f"util_parallel: {suffix_path} done")
        except:
            res = get_failed_obj()
            print(f"util_parallel: {suffix_path} failed")
        return res

    @classmethod
    def run_parallel(cls, params, command, max_proc=10):
        cls.shell(f"rm -rf {WKPATH}")
        param_vals = [p.values for p in params]
        param_vals_prod = itertools.product(*param_vals)
        job = functools.partial(cls.exec_single_job, params, command)
        with Pool(min(cpu_count(), max_proc)) as p:
            res = p.map(job, param_vals_prod)
        return cls.inverse_itertools_kd_product(param_vals, res)

if __name__ == "__main__":
    main_parallel()

