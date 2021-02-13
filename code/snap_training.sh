#!/bin/bash 

for BATCH in 2048 1024 512 256 128 64 32 16 8 4 2 1
do
	python3 soap_training.py ../data/ ../out/ --augment_steps= --nmax=9 --lmax=3 --batch_size=$BATCH
	python3 soap_training.py ../data/ ../out/ --augment_steps= --nmax=9 --lmax=3 --batch_size=$BATCH
	python3 soap_training.py ../data/ ../out/ --augment_steps= --nmax=9 --lmax=3 --batch_size=$BATCH
	python3 soap_training.py ../data/ ../out/ --augment_steps= --nmax=9 --lmax=3 --batch_size=$BATCH
done