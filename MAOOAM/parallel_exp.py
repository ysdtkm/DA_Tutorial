#!/usr/bin/env python3
# ref: a49p76

import sys, os
import subprocess as sp
import itertools
from multiprocessing import Pool, cpu_count
import numpy as np

def shell(cmd):
    assert isinstance(cmd, str)
    sp.run(cmd, shell=True, check=True)

def exec_single_job(param):
    p1, p2 = param
    dname = get_dir_name(p1, p2)
    shell("mkdir -p %s" % dname)
    os.chdir(dname)
    modeldir = "/home/tak/prgm/DA_Tutorial/MAOOAM"
    shell("cp -r %s/* . && make" % modeldir)
    print("%s done")
    return None

def get_dir_name(p1, p2):
    dn = "%s/oerr_%.05f/aint_%05d" % (WDIR_BASE, p1, p2)
    dn = dn.replace(".", "_")
    return dn

def reduce_mapped_result(params1, params2, map_res):
    import mydebug
    import matplotlib
    matplotlib.use("PDF")
    import matplotlib.pyplot as plt
    mydebug.dump_array(np.array(map_res))
    cm = plt.imshow(map_res)
    plt.colorbar(cm)
    plt.savefig("tmp.pdf")
    plt.close()

def inverse_itertools_2d_product(params1, params2, map_result):
    # un-flatten an iterable object made by itertools.product
    n1 = len(params1)
    n2 = len(params2)
    res = [[map_result[i1 * n2 + i2] for i2 in range(n2)] for i1 in range(n1)]
    return res

def main():
    global WDIR_BASE
    WDIR_BASE = "/home/tak/shrt/parallel_exp"

    params1 = [0.1 ** i for i in range(0, 4)]
    params2 = [10 ** i for i in range(0, 3)]
    params_prod = itertools.product(params1, params2)

    shell("rm -rf %s" % WDIR_BASE)
    with Pool(cpu_count()) as p:
        res = p.map(exec_single_job, params_prod)
    res2d = inverse_itertools_2d_product(params1, params2, res)
    reduce_mapped_result(params1, params2, res2d)

if __name__ == "__main__":
    main()

