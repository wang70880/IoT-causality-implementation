#!/bin/sh
partition=10
bk=1
mpiexec -n 35 python -u run_pcmci_parallel.py $partition $bk &> output.txt
