#!/usr/bin/env python
# coding: utf-8

import os
import sys
import time

#sys.path.insert(1, os.path.join("C:\\", "Users", "Soham", "Git", "spectral-NN", "source_codes"))
sys.path.insert(1, os.path.join("/home", "soham", "Git", "spectral-NN", "source_codes"))

from spectral_NN_multirun import spectral_NN_multirun as multirun

print(time.ctime())

method = "Shallow"
M = 10
L = 10
depth = None
width = 20
q = 20
multirun(method,M,L,depth,width,q)

print(time.ctime())

method = "Deep"
M = 10
L = 10
depth = 4
width = 20
q = 20
multirun(method,M,L,depth,width,q)

print(time.ctime())

method = "Deepshared1"
M = 10
L = 10
depth = 4
width = 20
q = 20
multirun(method,M,L,depth,width,q)

print(time.ctime())

method = "Deepshared2"
M = 10
L = 10
depth = 4
width = 20
q = 20
multirun(method,M,L,depth,width,q)

print(time.ctime())

method = "Deepshared3"
M = 10
L = 10
depth = 4
width = 20
q = 20
multirun(method,M,L,depth,width,q)

print(time.ctime())
