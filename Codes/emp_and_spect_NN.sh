#!/bin/bash

dim="3D"
cov_type="Matern"
par=0.001
gr_size=15
gam=0.5

N=1600
echo `date`
echo "${dim} ${cov_type}_${par} N=${N}, K=${gr_size}, gamma=${gam}"
repl=11
while [ ${repl} -ne 26 ];
do
	echo "Spectral-NN estimator"
	python Codes/spect_NN.py ${repl}
	echo `date`
	echo "Empirical spectral density estimator"
	python Codes/empirical.py ${repl}
	echo `date`
	repl=$((${repl}+1))
done
python Codes/cleanup.py ${dim} "N" ${N} ${gr_size} ${cov_type} ${par} ${gam}


for N in 800 400 200 100;
do
	rm -r "Data"
	mkdir "Data"
	echo `date`
	echo "${dim} ${cov_type}_${par} N=${N}, K=${gr_size}, gamma=${gam}"
	python Codes/datagen.py ${dim} ${N} ${gr_size} ${cov_type} ${par} ${gam}
	echo `date`
	cp -r "Data" "Data_Matern_${par}_3D_N=${N}_K=${gr_size}"
	repl=1
	while [ ${repl} -ne 26 ];
	do
		echo "Spectral-NN estimator"
		python Codes/spect_NN.py ${repl}
		echo `date`
		echo "Empirical spectral density estimator"
		python Codes/empirical.py ${repl}
		echo `date`
		repl=$((${repl}+1))
	done
	python Codes/cleanup.py ${dim} "N" ${N} ${gr_size} ${cov_type} ${par} ${gam}
done

