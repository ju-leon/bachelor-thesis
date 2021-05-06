#!/bin/bash 

for BATCH in 2048 512 128 16
do
	python3 soap_training.py ../data/ ../out/ --nmax=$1 --lmax=$2 --batch_size=$BATCH
done