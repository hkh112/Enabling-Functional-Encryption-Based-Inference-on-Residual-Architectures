#!/bin/bash

python -u test_tinet.py --terms=2 --unknown=16 --batch-size=1 --sife-l=64 --cusin=1 --test-num=50 2>&1 | tee "output_tinet_poly_term2_unknown16_l64_cusin1_time.txt"
python -u test_tinet.py --terms=2 --unknown=16 --batch-size=1 --sife-l=128 --cusin=2 --test-num=50 2>&1 | tee "output_tinet_poly_term2_unknown16_l128_cusin2_time.txt"
python -u test_tinet.py --terms=2 --unknown=16 --batch-size=1 --sife-l=128 --cusin=3 --test-num=50 2>&1 | tee "output_tinet_poly_term2_unknown16_l128_cusin3_time.txt"
python -u test_tinet.py --terms=2 --unknown=16 --batch-size=1 --sife-l=128 --cusin=4 --test-num=50 2>&1 | tee "output_tinet_poly_term2_unknown16_l128_cusin4_time.txt"

python -u test_tinet.py --terms=2 --unknown=16 --batch-size=1 --sife-l=64 --cusin=1 2>&1 | tee "output_tinet_poly_term2_unknown16_l64_cusin1.txt"