#!/bin/bash 

for NMAX in 2 3 4 5 6 7
do
    for LMAX in 0 1 2 3 4
    do
        python3 soap_parameter_test.py ../data/ ../out/ --test_split=0.1 --nmax=$NMAX --lmax=$LMAX
        python3 soap_parameter_test.py ../data/ ../out/ --test_split=0.2 --nmax=$NMAX --lmax=$LMAX
        python3 soap_parameter_test.py ../data/ ../out/ --test_split=0.3 --nmax=$NMAX --lmax=$LMAX
        python3 soap_parameter_test.py ../data/ ../out/ --test_split=0.4 --nmax=$NMAX --lmax=$LMAX
        python3 soap_parameter_test.py ../data/ ../out/ --test_split=0.5 --nmax=$NMAX --lmax=$LMAX
        python3 soap_parameter_test.py ../data/ ../out/ --test_split=0.6 --nmax=$NMAX --lmax=$LMAX
        python3 soap_parameter_test.py ../data/ ../out/ --test_split=0.7 --nmax=$NMAX --lmax=$LMAX
        python3 soap_parameter_test.py ../data/ ../out/ --test_split=0.8 --nmax=$NMAX --lmax=$LMAX
        python3 soap_parameter_test.py ../data/ ../out/ --test_split=0.9 --nmax=$NMAX --lmax=$LMAX
    done
done