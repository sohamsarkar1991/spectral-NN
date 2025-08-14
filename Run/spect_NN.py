#!/usr/bin/env python
# coding: utf-8

import os, sys

sys.path.insert(1, os.path.join("/home", "soham", "Git", "spectral-NN", "source_codes"))

from spectral_NN_multirun import spectral_NN_multirun as spect_multirun

if __name__ == "__main__":
    if len(sys.argv) > 1:
        repl = int(sys.argv[1])
        try:
            spect_multirun("Deep",M=10,L=10,depth=4,width=20,q=20,replicates=range(repl-1,repl))
        except Exception as e:
            print("Couldn't fit spectral network due to some error. More details in 'spectral_NN_issues.txt'")
            f = open(os.path.join("Results","spectral_NN_issues.txt"),"a")
            f.write('{}' .format(e))
            f.close()
