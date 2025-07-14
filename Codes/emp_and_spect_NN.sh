#!/bin/bash

echo `date`

dim="3D"
echo "Simulations in ${dim}:"

N=250
gr_size=25
gam=0.5
for method in "BM,None" "IBM,None" "Matern,0.001" "Matern,0.01" "Matern,0.1" "Matern,1";
do
	IFS=',' read -r cov_type par <<< "${method}"
	echo `date`
	if [ ${cov_type} == "Matern" ]
	then
		echo "${cov_type} ${par}:"
		python Codes/datagen.py ${dim} ${N} ${gr_size} ${cov_type} ${par} ${gam}
	else
		echo "${cov_type}:"
		par="None"
		python Codes/datagen.py ${dim} ${N} ${gr_size} ${cov_type} ${par} ${gam}
	fi
	echo `date`
	echo "Spectral-NN estimator:"
	python Codes/spect_NN.py
	echo `date`
	echo "Empirical spectral density estimator:"
	repl=1
	while [ ${repl} -ne 26 ];
	do
		#echo "${repl}"
		python Codes/empirical.py ${repl}
		repl=$((${repl}+1))
	done
	python Codes/cleanup.py "gr_size" ${N} ${gr_size} ${cov_type} ${par} ${gam}
done


