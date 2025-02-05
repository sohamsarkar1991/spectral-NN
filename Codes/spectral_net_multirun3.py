#!/usr/bin/env python
# coding: utf-8

import os
import sys
import time

#sys.path.insert(1, os.path.join("C:\\", "Users", "Soham", "Git", "spectral-NN", "source_codes"))
sys.path.insert(1, os.path.join("/home", "soham", "Git", "spectral-NN", "source_codes"))

from spectral_NN_multirun import spectral_NN_multirun as multirun

print(time.ctime())

method = "Deep"

for L in [5,10]:
    multirun(method,M=20,L=L,depth=4,width=20,q=20)
    print(time.ctime())

