#!/bin/bash 

for ASTEP in 1 5 10 12 16 20 25 30 50 100
do
    python3 soap_training.py ../data/ ../out/ --augment_steps=$ASTEP
done