#!/bin/bash 


for RCUT in 12 20 8
do
	for NMAX in 2 3 4 5 6 7 8 9
	do
    		for LMAX in 2 3 4 5 6 7 8 9
    		do
        		python3 soap_training.py ../data/ ../out_$1/ --augment_steps=$1 --nmax=$NMAX --lmax=$LMAX
    		done
	done
done
