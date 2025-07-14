#!/bin/bash

echo `date`

: '
dim="1D"
echo "Simulations in ${dim}:"

gr_size=200
#gr_size=50
#gr_size=15
for N in 100 200 400 800 1600;
do
	echo "N = $N, Grid size = $gr_size"
	python Memory_usage/cleanup.py
	python Memory_usage/datagen.py ${dim} ${N} ${gr_size}
	filename="Memory_usage/${dim}/Empirical_${N}_${gr_size}"
	echo "Empirical spectral density estimator"
	scalene --cli --outfile ${filename} --html --reduced-profile Memory_usage/empirical.py
	filename="Memory_usage/${dim}/Spectral_${N}_${gr_size}"
	echo "Spectral-NN estimator"
	scalene --cli --outfile ${filename} --html --reduced-profile Memory_usage/spect_NN.py
	echo `date`
done

N=250
for gr_size in 20 40 80 160 320 640 1280 2560; 
#for gr_size in 10 20 40 80 160 320;
#for gr_size in 10 15 20 25 30;
do
	echo "N = $N, Grid size = $gr_size"
	python Memory_usage/cleanup.py
	python Memory_usage/datagen.py ${dim} ${N} ${gr_size}
	filename="Memory_usage/${dim}/Empirical_${N}_${gr_size}"
	echo "Empirical spectral density estimator"
	scalene --cli --outfile ${filename} --html --reduced-profile Memory_usage/empirical.py
	filename="Memory_usage/${dim}/Spectral_${N}_${gr_size}"
	echo "Spectral-NN estimator"
	scalene --cli --outfile ${filename} --html --reduced-profile Memory_usage/spect_NN.py
	echo `date`
done
'

for par in "1D,200,200" "1D,800,200" "1D,250,160" "2D,800,50" "3D,1600,15";
do
	IFS=',' read -r dim N gr_size <<< "${par}"
	echo "${dim} Simulations: N = ${N}, Grid size = ${gr_size}"
	python Memory_usage/cleanup.py
	python Memory_usage/datagen.py ${dim} ${N} ${gr_size}
	filename="Memory_usage/${dim}/Empirical_${N}_${gr_size}"
	echo "Empirical spectral density estimator"
	scalene --cli --outfile ${filename} --html --reduced-profile Memory_usage/empirical.py
	filename="Memory_usage/${dim}/Spectral_${N}_${gr_size}"
	echo "Spectral-NN estimator"
	scalene --cli --outfile ${filename} --html --reduced-profile Memory_usage/spect_NN.py
	echo `date`
done
