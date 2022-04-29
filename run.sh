#!/bin/bash
mpiexec -n 35 python -u run_pcmci_parallel.py &> output.txt
