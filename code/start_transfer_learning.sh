#!/bin/bash

mkdir -p $1

python3 transfer_learning.py ../data/morpheus/morfeus_properties/ $1 --nmax=$2 --lmax=$3 --ligand_test=$4 --dataset=full_transfer
python3 transfer_learning.py ../data/morpheus/morfeus_properties/ $1 --nmax=$2 --lmax=$3 --ligand_test=$4 --dataset=vask_transfer

python3 transfer_learning_adapt.py ../data/ $1 --nmax=$2 --lmax=$3 --ligand_test=$4 --dataset=no_transfer
python3 transfer_learning_adapt.py ../data/ $1 --nmax=$2 --lmax=$3 --ligand_test=$4 --dataset=full_transfer
python3 transfer_learning_adapt.py ../data/ $1 --nmax=$2 --lmax=$3 --ligand_test=$4 --dataset=vask_transfer
