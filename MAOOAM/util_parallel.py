#!/usr/bin/env python3

import sys, os
import subprocess as sp
import itertools, functools
from multiprocessing import Pool, cpu_count
import numpy as np

class Change:
    def __init__(self, file_name, line_num, match, replace_fmt):
        assert isinstance(file_name, str)
        assert isinstance(line_num, int)
        assert isinstance(match, str)
        assert isinstance(replace_fmt, str)
        self.file_name = file_name
        self.line_num = line_num
        self.match = match
        self.replace_fmt = replace_fmt

    def rewrite_file_with_param(self, param):
        with open(self.file_name, "r") as f:
            lines = f.readlines()
        with open(self.file_name, "w") as f:
            for i, l in enumerate(lines):
                if i == self.line_num - 1:
                    assert self.match in l
                    f.write(self.replace_fmt % param + "\n")
                else:
                    f.write(l)

def inverse_itertools_2d_product(params1, params2, map_result):
    # return 2D list, un-flattening the map_result
    n1 = len(params1)
    n2 = len(params2)
    res = [[map_result[i1 * n2 + i2] for i2 in range(n2)] for i1 in range(n1)]
    return res

def shell(cmd, check=True):
    assert isinstance(cmd, str)
    p = sp.run(cmd, shell=True, encoding="utf8", stderr=sp.PIPE, stdout=sp.PIPE)
    if check and p.returncode != 0:
        print(p.stdout)
        print(p.stderr)
        raise Exception("shell %s failed" % cmd)
    return p.stdout, p.stderr

def exec_parallel(dir_template, wdir_base, params1, params2, p1_fmt, p2_fmt,
                  p1_changes, p2_changes, command, out_file):
    shell("mkdir -p %s/out" % wdir_base)
    params_prod = itertools.product(params1, params2)
    job = functools.partial(exec_single_job, dir_template, wdir_base, p1_fmt, p2_fmt,
                            p1_changes, p2_changes, command, out_file)
    with Pool(5) as p:
        res = p.map(job, params_prod)
    return inverse_itertools_2d_product(params1, params2, res)

def exec_single_job(wdir_base, dir_template, p1_fmt, p2_fmt, p1_changes, p2_changes,
                    command, out_file, param):
    p1, p2 = param
    s1 = (p1_fmt % p1).replace(".", "_")
    s2 = (p2_fmt % p2).replace(".", "_")
    dname = "%s/%s/%s" % (wdir_base, s1, s2)
    shell("mkdir -p %s" % dname)
    os.chdir(dname)
    shell("cp -rf %s/%s/* ." % (wdir_base, dir_template))
    for c1 in p1_changes:
        c1.rewrite_file_with_param(p1)
    for c2 in p2_changes:
        c2.rewrite_file_with_param(p2)
    sout, serr = shell(command, check=False)
    with open("stdout", "w") as f:
        f.write(sout)
    with open("stderr", "w") as f:
        f.write(serr)
    _, ext = os.path.splitext(out_file)
    try:
        shell("cp -f %s %s/out/%s_%s%s" % (out_file, wdir_base, s1, s2, ext))
        res = np.load("rmse_hybrid.npy")
        print("%s done" % dname)
    except:
        res = np.empty(5)
        res[:] = np.nan
        print("%s failed" % dname)
    return res

