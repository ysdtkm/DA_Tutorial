#!/usr/bin/env python3

import sys
from util_parallel import Change, shell, exec_parallel

def main():
    wdir_base = sys.argv[1]
    params1 = list(reversed(range(3, 38, 2)))
    params2 = [1.0 + 0.01 * i for i in range(6)]
    changes1 = [Change("analysis_init.py", 79, "das.edim", "das.edim = %d")]
    changes2 = [Change("class_da_system.py", 420, "rho", "    rho = %f")]
    shell("mkdir -p %s/out" % wdir_base)
    exec_parallel(wdir_base, "template", params1, params2, "ens_%02d", "infl_%.02f",
                  changes1, changes2, "make", "out.pdf")

if __name__ == "__main__":
    main()

