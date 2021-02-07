#!/bin/bash 

for ASTEP in 5 10 20 50 100
do
    for NMAX in 2 3 4 5 6 7 8 9
    do
        for LMAX in 2 3 4 5 6 7 8 9
        do
            python3 soap_training.py ../data/ ../out/ --augment_steps=$ASTEP --nmax=$NMAX --lmax=$LMAX
        done
    done
done