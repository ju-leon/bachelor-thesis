#!/bin/bash 

for BATCH in 2048 
do
	python3 soap_training.py ../data/ ../out/ --nmax=$1 --lmax=$2 --batch_size=$BATCH --interpolation_steps=$3 --name=$4 --sidegroup_validation=$5
done
