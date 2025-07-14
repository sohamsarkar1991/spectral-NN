#!/usr/bin/env python
# coding: utf-8

import sys, os

sys.path.insert(1, os.path.join("/home", "soham", "Git", "spectral-NN", "source_codes"))

from empirical_spectral_density_multirun import emp_spect_dens_multirun as emp_multirun

try:
    emp_multirun(q=20,replicates=range(1))
except Exception as e:
    print("Couldn't compute empirical spectral density.")

