#!/usr/bin/env python3
# ref: a49p76

import sys, os, re
import subprocess as sp
import itertools
from multiprocessing import Pool, cpu_count
import numpy as np

class GlobalParams:
    wdir_base = "/lustre/tyoshida/shrt/exec/parallel_exp"
    params1 = [0.1 ** i for i in range(4, 5)]
    params2 = list(range(2, 38))

    def get_wdir_absolute_path(p1, p2):
        assert p1 in GlobalParams.params1
        assert p2 in GlobalParams.params2
        dn = "%s/oerr_%.05f/aint_%05d" % (GlobalParams.wdir_base, p1, p2)
        dn = dn.replace(".", "_")
        return dn

    def get_relative_file_name(p1, p2):
        dn = "oerr_%.05f_aint_%05d" % (p1, p2)
        dn = dn.replace(".", "_")
        return dn

    def get_params_prod():
        return itertools.product(GlobalParams.params1, GlobalParams.params2)

    def inverse_itertools_2d_product(map_result):
        # return 2D list, un-flattening the map_result
        try:
            dummy = iter(map_result)
        except:
            raise Exception
        n1 = len(GlobalParams.params1)
        n2 = len(GlobalParams.params2)
        res = [[map_result[i1 * n2 + i2] for i2 in range(n2)] for i1 in range(n1)]
        return res

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
    dname = GlobalParams.get_wdir_absolute_path(p1, p2)
    shell("mkdir -p %s" % dname)
    os.chdir(dname)
    shell("cp -r %s/template/* ." % GlobalParams.wdir_base)
    rewrite_file("analysis_init.py", 79, "das.edim", "das.edim = %d" % p2)
    rewrite_file("analysis_init.py", 104, "sigma_r", "sigma_r = %f" % p1)
    rewrite_file("generate_observations.py", 14, "sigma", "sigma = %f" % p1)
    sout, serr = shell("make")
    with open("stdout", "w") as f:
        f.write(sout)
    with open("stderr", "w") as f:
        f.write(serr)
    shell("cp -f out.pdf %s/out/%s.pdf" %
        (GlobalParams.wdir_base, GlobalParams.get_relative_file_name(p1, p2)))
    print("%s done" % dname)

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

def main():
    if len(sys.argv) > 1:
        GlobalParams.wdir_base = sys.argv[1]
    shell("mkdir -p %s/out" % GlobalParams.wdir_base)
    params_prod = GlobalParams.get_params_prod()
    with Pool(cpu_count()) as p:
        res = p.map(exec_single_job, params_prod)
    res2d = GlobalParams.inverse_itertools_2d_product(res)

if __name__ == "__main__":
    main()

