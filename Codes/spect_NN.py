#!/usr/bin/env python
# coding: utf-8

import os, sys, time, shutil

sys.path.insert(1, os.path.join("/home", "soham", "Git", "spectral-NN", "source_codes"))

from spectral_NN_multirun import spectral_NN_multirun as spect_multirun

print("Spectral-NN model:\n")
try:
    spect_multirun("Deep",M=10,L=10,depth=4,width=20,q=20,replicates=range(25))
except Exception as e:
    print("Couldn't fit spectral network due to some error. More details in 'spectral_NN_issues.txt'")
    f = open(os.path.join("Results","spectral_NN_issues.txt"),"a")
    f.write('{}' .format(e))
    f.close()
