#!/bin/bash

dim="3D"
cov_type="Matern"
par=0.001
N=250
gam=0.5

for gr_size in 10 15 20 25 30;
do
	echo `date`
	echo "${dim} ${cov_type}_${par} N=${N}, K=${gr_size}, gamma=${gam}"
	python Run/datagen.py ${dim} ${N} ${gr_size} ${cov_type} ${par} ${gam}
	cp -r "Data" "Data_Matern_${par}_3D_N=${N}_K=${gr_size}"
	echo `date`
	repl=1
	while [ ${repl} -ne 26 ];
	do
		echo "Spectral-NN estimator"
		python Run/spect_NN_gpu.py ${repl}
		echo `date`
		echo "Empirical spectral density estimator"
		python Run/empirical.py ${repl}
		echo `date`
		repl=$((${repl}+1))
	done
	python Run/cleanup.py ${dim} "N" ${N} ${gr_size} ${cov_type} ${par} ${gam}
done

