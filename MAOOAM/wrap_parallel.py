#!/usr/bin/env python3
# ref: a49p76

import sys, os, re
import subprocess as sp
import itertools
from multiprocessing import Pool, cpu_count
import numpy as np

_wdir_base = "/home/tak/shrt/parallel_exp"

def shell(cmd):
    assert isinstance(cmd, str)
    p = sp.run(cmd, shell=True, encoding="utf8", stderr=sp.PIPE, stdout=sp.PIPE)
    if p.returncode != 0:
        print(p.stdout)
        print(p.stderr)
        raise Exception("shell %s failed" % cmd)
    return p.stdout, p.stderr

def exec_single_job(param):
    p1, p2 = param
    dname = get_dir_name(p1, p2)
    shell("mkdir -p %s" % dname)
    os.chdir(dname)
    shell("cp -r %s/template/* ." % _wdir_base)
    rewrite_file("analysis_init.py", 137, "acyc_step", "acyc_step = %d" % p2)
    rewrite_file("analysis_init.py", 104, "sigma_r", "sigma_r = %f" % p1)
    rewrite_file("generate_observations.py", 14, "sigma", "sigma = %f" % p1)
    shell("make")
    shell("cp -f out.pdf %s/out/%s.pdf" % (_wdir_base, get_file_name(p1, p2)))
    print("%s done" % dname)
    return None

def rewrite_file(filename, linenum, txt_from, txt_to):
    with open(filename, "r") as f:
        lines = f.readlines()
    with open(filename, "w") as f:
        for i, l in enumerate(lines):
            if i == linenum - 1:
                assert txt_from in l
                f.write(txt_to + "\n")
            else:
                f.write(l)

def get_dir_name(p1, p2):
    dn = "%s/oerr_%.05f/aint_%05d" % (_wdir_base, p1, p2)
    dn = dn.replace(".", "_")
    return dn

def get_file_name(p1, p2):
    dn = "oerr_%.05f_aint_%05d" % (p1, p2)
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
    global _wdir_base
    if len(sys.argv) > 1:
        _wdir_base = sys.argv[1]
    shell("mkdir -p %s/out" % _wdir_base)

    params1 = [0.1 ** i for i in range(0, 2)]
    params2 = [10 ** i for i in range(0, 2)]
    params_prod = itertools.product(params1, params2)

    with Pool(cpu_count()) as p:
        res = p.map(exec_single_job, params_prod)
    res2d = inverse_itertools_2d_product(params1, params2, res)
    # reduce_mapped_result(params1, params2, res2d)

if __name__ == "__main__":
    main()

