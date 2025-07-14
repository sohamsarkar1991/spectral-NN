#!/usr/bin/env python
# coding: utf-8

import os, sys, time, shutil

sys.path.insert(1, os.path.join("/home", "soham", "Git", "spectral-NN", "source_codes"))

from spectral_NN_multirun import spectral_NN_multirun as spect_multirun

try:
    spect_multirun("Deep",M=10,L=10,depth=4,width=20,q=20,replicates=range(1))
except Exception as e:
    print("Couldn't fit spectral network.")

