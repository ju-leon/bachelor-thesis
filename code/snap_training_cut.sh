#!/bin/bash 

for NMAX in 2 3 4 5 6 7 8 9
do
    for LMAX in 2 3 4 5 6 7 8 9
    do
        python3 snap_cutoff_training.py ../data/ ../out/ --augment_steps=$1 --del_rad=$2 --nmax=$NMAX --lmax=$LMAX
    done
done
