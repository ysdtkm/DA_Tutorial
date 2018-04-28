#!/usr/bin/env python3

import sys
from util_parallel import Change, shell, exec_parallel

def main():
    wdir_base = sys.argv[1]
    params1 = [1.0 + 0.01 * i for i in range(11)]
    params2 = list(range(3, 38, 2))
    changes1 = [Change("class_da_system.py", 420, "rho", "    rho = %f")]
    changes2 = [Change("analysis_init.py", 79, "das.edim", "das.edim = %d")]
    shell("mkdir -p %s/out" % wdir_base)
    exec_parallel(wdir_base, "template", params1, params2, "infl_%.02f", "ens_%02d",
                  changes1, changes2, "make", "out.pdf")

if __name__ == "__main__":
    main()

