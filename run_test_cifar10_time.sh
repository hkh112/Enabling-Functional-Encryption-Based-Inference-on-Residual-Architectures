#!/bin/bash

python -u test_cifar10.py --terms=2 --unknown=16 --batch-size=1 --sife-l=128 --cusin=2 2>&1 | tee "output_cifar10_poly_term2_unknown16_l128_cusin2.txt"
python -u test_cifar10.py --terms=2 --unknown=16 --batch-size=1 --sife-l=128 --cusin=3 2>&1 | tee "output_cifar10_poly_term2_unknown16_l128_cusin3.txt"
python -u test_cifar10.py --terms=2 --unknown=16 --batch-size=1 --sife-l=128 --cusin=4 2>&1 | tee "output_cifar10_poly_term2_unknown16_l128_cusin4.txt"

