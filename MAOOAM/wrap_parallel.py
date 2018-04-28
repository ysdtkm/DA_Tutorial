#!/usr/bin/env python3

import sys
from util_parallel import Change, shell, exec_parallel

def main():
    wdir_base = sys.argv[1]
    params1 = [1.0 + 0.01 * i for i in range(2)]
    params2 = list(range(8, 38, 15))
    changes1 = []
    changes2 = []
    shell("mkdir -p %s/out" % wdir_base)
    exec_parallel(wdir_base, "template", params1, params2, "infl_%.02f", "ens_%02d",
                  changes1, changes2, "make", "out.pdf")

if __name__ == "__main__":
    main()

