#!/bin/bash 

python3 soap_training.py ../data/ ../out/ --augment_steps=20 --rcut=10 --nmax=6 --lmax=3
python3 soap_training.py ../data/ ../out/ --augment_steps=20 --rcut=10 --nmax=7 --lmax=3
python3 soap_training.py ../data/ ../out/ --augment_steps=20 --rcut=20 --nmax=6 --lmax=3
python3 soap_training.py ../data/ ../out/ --augment_steps=20 --rcut=20 --nmax=7 --lmax=3

python3 soap_training.py ../data/ ../out/ --augment_steps=30 --rcut=10 --nmax=6 --lmax=3
python3 soap_training.py ../data/ ../out/ --augment_steps=30 --rcut=10 --nmax=7 --lmax=3
python3 soap_training.py ../data/ ../out/ --augment_steps=30 --rcut=20 --nmax=6 --lmax=3
python3 soap_training.py ../data/ ../out/ --augment_steps=30 --rcut=20 --nmax=7 --lmax=3

python3 soap_training.py ../data/ ../out/ --augment_steps=40 --rcut=10 --nmax=6 --lmax=3
python3 soap_training.py ../data/ ../out/ --augment_steps=40 --rcut=10 --nmax=7 --lmax=3
python3 soap_training.py ../data/ ../out/ --augment_steps=40 --rcut=20 --nmax=6 --lmax=3
python3 soap_training.py ../data/ ../out/ --augment_steps=40 --rcut=20 --nmax=7 --lmax=3

python3 soap_training.py ../data/ ../out/ --augment_steps=60 --rcut=10 --nmax=6 --lmax=3
python3 soap_training.py ../data/ ../out/ --augment_steps=60 --rcut=10 --nmax=7 --lmax=3
python3 soap_training.py ../data/ ../out/ --augment_steps=60 --rcut=20 --nmax=6 --lmax=3
python3 soap_training.py ../data/ ../out/ --augment_steps=60 --rcut=20 --nmax=7 --lmax=3

python3 soap_training.py ../data/ ../out/ --augment_steps=100 --rcut=10 --nmax=6 --lmax=3
python3 soap_training.py ../data/ ../out/ --augment_steps=100 --rcut=10 --nmax=7 --lmax=3
python3 soap_training.py ../data/ ../out/ --augment_steps=100 --rcut=20 --nmax=6 --lmax=3
python3 soap_training.py ../data/ ../out/ --augment_steps=100 --rcut=20 --nmax=7 --lmax=3