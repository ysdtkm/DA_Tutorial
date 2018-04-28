#!/usr/bin/env python3
# ref: a49p76

import sys, os, re
import subprocess as sp
import itertools
from multiprocessing import Pool, cpu_count
import numpy as np

class GlobalParams:
    wdir_base = "/lustre/tyoshida/shrt/exec/parallel_exp"
    params1 = [1.0 + 0.01 * i for i in range(11)]
    params2 = list(range(2, 38, 6))  # ens

    def str_param1(p1):
        res = "rho_%.03f" % p1
        return res.replace(".", "_")

    def str_param2(p2):
        res = "ens_%05d" % p2
        return res

    def rewrite_files(p1, p2):
        rewrite_line("class_da_system.py", 420, "rho", "rho = %f" % p1)
        rewrite_line("analysis_init.py", 79, "das.edim", "das.edim = %d" % p2)

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
    dname = "%s/%s/%s" % (GlobalParams.wdir_base, GlobalParams.str_param1(p1), GlobalParams.str_param2(p2))
    shell("mkdir -p %s" % dname)
    os.chdir(dname)
    shell("cp -r %s/template/* ." % GlobalParams.wdir_base)
    GlobalParams.rewrite_files(p1, p2)
    sout, serr = shell("make")
    with open("stdout", "w") as f:
        f.write(sout)
    with open("stderr", "w") as f:
        f.write(serr)
    shell("cp -f out.pdf %s/out/%s_%s.pdf" %
        (GlobalParams.wdir_base, GlobalParams.str_param1(p1), GlobalParams.str_param2(p2)))
    print("%s done" % dname)

def rewrite_line(filename, linenum, txt_from, txt_to):
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

