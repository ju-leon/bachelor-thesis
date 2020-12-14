#!/bin/bash 

for RCUT in 3 4 5 6 7 8 9 10
do
    python3 soap_parameter_test.py ../data/ ../out/ --test_split=0.8 --rcut=$RCUT
done