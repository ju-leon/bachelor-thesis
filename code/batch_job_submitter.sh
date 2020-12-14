#!/bin/bash 

for RCUT in 11 12 13 14 15 16
do
    python3 soap_parameter_test.py ../data/ ../out/ --test_split=0.8 --rcut=$RCUT
done