#!/usr/bin/env python
# coding: utf-8

import sys, os

sys.path.insert(1, os.path.join("/home", "soham", "Git", "spectral-NN", "source_codes"))

from empirical_spectral_density_multirun import emp_spect_dens_multirun as emp_multirun

if __name__ == "__main__":
    if len(sys.argv) > 1:
        repl = int(sys.argv[1])
        try:
            emp_multirun(q=20,replicates=range(repl-1,repl))
        except Exception as e:
            print("Couldn't compute empirical spectral density due to some error. More details in 'empirical_issues.txt'")
            f = open(os.path.join("Results","empirical_issues.txt"),"a")
            f.write('{}' .format(e))
            f.close()

