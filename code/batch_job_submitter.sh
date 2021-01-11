#!/bin/bash 

for ASTEP in 1 3 5 8 10 12 16 20 50
do
    python3 soap_training.py ../data/ ../out/ --augment_steps=$ASTEP
done