#!/bin/bash

python -u test_tinet_polynomial.py --terms=1 --unknown=16 --batch-size=128 --sife-l=128 --cusin=1 2>&1 | tee "output_tinet_poly_term1_unknown16_l128_cusin1.txt"
python -u test_tinet_polynomial.py --terms=2 --unknown=16 --batch-size=128 --sife-l=128 --cusin=1 2>&1 | tee "output_tinet_poly_term2_unknown16_l128_cusin1.txt"
python -u test_tinet_polynomial.py --terms=3 --unknown=16 --batch-size=128 --sife-l=128 --cusin=1 2>&1 | tee "output_tinet_poly_term3_unknown16_l128_cusin1.txt"
python -u test_tinet_polynomial.py --terms=4 --unknown=16 --batch-size=128 --sife-l=128 --cusin=1 2>&1 | tee "output_tinet_poly_term4_unknown16_l128_cusin1.txt"

python -u test_tinet_polynomial.py --terms=1 --unknown=16 --batch-size=128 --sife-l=128 --cusin=2 2>&1 | tee "output_tinet_poly_term1_unknown16_l128_cusin2.txt"
python -u test_tinet_polynomial.py --terms=2 --unknown=16 --batch-size=128 --sife-l=128 --cusin=2 2>&1 | tee "output_tinet_poly_term2_unknown16_l128_cusin2.txt"
python -u test_tinet_polynomial.py --terms=3 --unknown=16 --batch-size=128 --sife-l=128 --cusin=2 2>&1 | tee "output_tinet_poly_term3_unknown16_l128_cusin2.txt"
python -u test_tinet_polynomial.py --terms=4 --unknown=16 --batch-size=128 --sife-l=128 --cusin=2 2>&1 | tee "output_tinet_poly_term4_unknown16_l128_cusin2.txt"

python -u test_tinet_polynomial.py --terms=1 --unknown=16 --batch-size=128 --sife-l=128 --cusin=3 2>&1 | tee "output_tinet_poly_term1_unknown16_l128_cusin3.txt"
python -u test_tinet_polynomial.py --terms=2 --unknown=16 --batch-size=128 --sife-l=128 --cusin=3 2>&1 | tee "output_tinet_poly_term2_unknown16_l128_cusin3.txt"
python -u test_tinet_polynomial.py --terms=3 --unknown=16 --batch-size=128 --sife-l=128 --cusin=3 2>&1 | tee "output_tinet_poly_term3_unknown16_l128_cusin3.txt"
python -u test_tinet_polynomial.py --terms=4 --unknown=16 --batch-size=128 --sife-l=128 --cusin=3 2>&1 | tee "output_tinet_poly_term4_unknown16_l128_cusin3.txt"

python -u test_tinet_polynomial.py --terms=1 --unknown=16 --batch-size=128 --sife-l=128 --cusin=4 2>&1 | tee "output_tinet_poly_term1_unknown16_l128_cusin4.txt"
python -u test_tinet_polynomial.py --terms=2 --unknown=16 --batch-size=128 --sife-l=128 --cusin=4 2>&1 | tee "output_tinet_poly_term2_unknown16_l128_cusin4.txt"
python -u test_tinet_polynomial.py --terms=3 --unknown=16 --batch-size=128 --sife-l=128 --cusin=4 2>&1 | tee "output_tinet_poly_term3_unknown16_l128_cusin4.txt"
python -u test_tinet_polynomial.py --terms=4 --unknown=16 --batch-size=128 --sife-l=128 --cusin=4 2>&1 | tee "output_tinet_poly_term4_unknown16_l128_cusin4.txt"