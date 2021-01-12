#!/bin/bash 

for ASTEP in 20 50 100 200 400 800 1000 5000
do
    python3 soap_training.py ../data/ ../out/ --augment_steps=$ASTEP
done