#!/bin/bash 

for BATCH in 2048 512 128 32 8 1
do
	python3 soap_training.py ../data/ ../out/ --augment_steps= --nmax=9 --lmax=3 --batch_size=$BATCH
	python3 soap_training.py ../data/ ../out/ --augment_steps= --nmax=9 --lmax=4 --batch_size=$BATCH
	python3 soap_training.py ../data/ ../out/ --augment_steps= --nmax=7 --lmax=3 --batch_size=$BATCH
	python3 soap_training.py ../data/ ../out/ --augment_steps= --nmax=8 --lmax=4 --batch_size=$BATCH
done