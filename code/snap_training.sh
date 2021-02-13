#!/bin/bash 

for BATCH in 2048 512 128 32 8 1
do
	python3 soap_training.py ../data/ ../out/ --augment_steps= --nmax=$1 --lmax=$2 --batch_size=$BATCH
done