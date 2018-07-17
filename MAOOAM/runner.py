#!/usr/bin/env python3

import collections
import functools
import itertools
from multiprocessing import Pool, cpu_count
import re
import os
import pickle
import sys
import subprocess as sp
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np

def main_parallel():
    wkpath = sys.argv[1]
    templatepath = f"{wkpath}/template"
    params = [
        ParameterAxis("ensemble member", list(range(2, 38, 12)), "%02d", \
            [Rewriter("exp_params.py", 16, "EDIM = {param:d}")])
    ]
    if True:
        res = Runner.run_parallel(params, "make", wkpath, templatepath)
        save_results(params, res)
    else:
        params, res = load_results()
    visualize_1d_rmse(params, res, wkpath)
    visualize_2d_rmse(params, res, wkpath)

def save_results(params, res):
    with open("all_results.pkl", "wb") as f:
        pickle.dump([params, res], f)

def load_results():
    with open("all_results.pkl", "rb") as f:
        params, res = pickle.load(f)
    return params, res

def visualize_1d_rmse(params, res, wkpath):
    assert len(params) == 1
    assert isinstance(res, np.ndarray)
    names = ["atmos_psi", "atmos_temp", "ocean_psi", "ocean_temp", "all"]
    for ic, component in enumerate(names):
        subtract_rmse = np.vectorize(lambda x: x[ic])
        plt.semilogy(params[0].values, subtract_rmse(res))
        plt.title(f"RMSE of {component} variables")
        plt.xlabel(params[0].name)
        plt.ylabel("RMS error")
        plt.savefig(f"{wkpath}/out/rmse_onedim_{component}.pdf", bbox_inches="tight")
        plt.close()

def visualize_2d_rmse(params, res, wkpath):
    if len(params) == 1:
        params.append(ParameterAxis("none", [0], "%01d", []))
        res = res[:, None]
    assert len(params) == 2
    names = ["atmos_psi", "atmos_temp", "ocean_psi", "ocean_temp", "all"]
    for ic, component in enumerate(names):
        fig, ax = plt.subplots()
        subtract_rmse = np.vectorize(lambda x: x[ic])
        cm = plt.imshow(subtract_rmse(res), cmap="Reds", norm=matplotlib.colors.LogNorm())
        plt.colorbar(cm)
        plt.title(f"RMSE of {component} variables")
        plt.xlabel(params[1].name)
        ax.set_xticks(range(len(params[1].values)))
        ax.set_xticklabels(params[1].values, rotation=90)
        plt.ylabel(params[0].name)
        ax.set_yticks(range(len(params[0].values)))
        ax.set_yticklabels(params[0].values)
        plt.savefig(f"{wkpath}/out/rmse_{component}.pdf", bbox_inches="tight")
        plt.close()

def get_output_obj(suffix_path, wkpath):
    fname = suffix_path.replace("/", "_")[1:]
    shell(f"cp -f out.pdf {wkpath}/out/{fname}.pdf")
    npa = np.load("rmse_ETKF.npy")
    return npa

def get_failed_obj(suffix_path, wkpath):
    return np.ones(5) * np.nan

# Following are backend. Generally no need to edit.
class Rewriter:
    def __init__(self, relative_file_name, line_num, formattable_replacer, disable_check=False):
        assert isinstance(relative_file_name, str)
        assert isinstance(line_num, int)
        assert isinstance(formattable_replacer, str)
        self.relative_file_name = relative_file_name
        self.line_num = line_num
        self.formattable_replacer = formattable_replacer
        self.disable_check = disable_check

    def rewrite_file_with_param(self, param):
        with open(self.relative_file_name, "r") as f:
            lines = f.readlines()
        str_to_match = re.sub(r"\{[^)]*\}", "", self.formattable_replacer)  # remove curly brackets
        with open(self.relative_file_name, "w") as f:
            for i, l in enumerate(lines):
                if i == self.line_num - 1:
                    if (not self.disable_check) and (not str_to_match in l):
                        raise ValueError(f"Rewriter checking error: pattern '{str_to_match}' not in line {self.line_num} of {self.relative_file_name}")
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
    def exec_single_job(cls, params, command, wkpath, templatepath, list_param_val):
        k_dim = len(params)
        assert len(list_param_val) == k_dim
        str_path_part = [(params[j].path_fmt % list_param_val[j]).replace(".", "_") for j in range(k_dim)]
        suffix_path = "".join([f"/{str_path_part[j]}" for j in range(k_dim)])
        single_dir_name = f"{wkpath}{suffix_path}"
        shell(f"mkdir -p {single_dir_name}")
        os.chdir(single_dir_name)
        shell(f"cp -rf {templatepath}/* .")
        for j in range(k_dim):
            for r in params[j].rewriters:
                r.rewrite_file_with_param(list_param_val[j])
        try:
            shell(command, writeout=True)
            res = get_output_obj(suffix_path, wkpath)
            print(f"util_parallel: {suffix_path} done")
        except:
            res = get_failed_obj(suffix_path, wkpath)
            print(f"util_parallel: {suffix_path} failed")
        return res

    @classmethod
    def run_parallel(cls, params, command, wkpath, templatepath, max_proc=10):
        shell(f"rm -rf {wkpath}/out")
        shell(f"mkdir -p {wkpath}/out")
        param_vals = [p.values for p in params]
        param_vals_prod = itertools.product(*param_vals)
        job = functools.partial(cls.exec_single_job, params, command, wkpath, templatepath)
        with Pool(min(cpu_count(), max_proc)) as p:
            res = p.map(job, param_vals_prod)
        return cls.inverse_itertools_kd_product(param_vals, res)

def shell(cmd, writeout=False):
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

if __name__ == "__main__":
    main_parallel()

