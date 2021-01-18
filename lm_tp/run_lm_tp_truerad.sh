#!/usr/bin/env bash

for l in {1..500}; do mkdir $l; done
for l in {1..500}; do cp hd_lm_tp_truerad_supp.py hd_lm_tp_truerad_supp_$l.py; sed -i "s/l = 1/l = $l/g" "hd_lm_tp_truerad_supp_$l.py"; done
for l in {1..500}; do cp hd_lm_tp_truerad.sub hd_lm_tp_truerad_$l.sub; sed -i "s/hd_lm_tp_truerad_supp.py/hd_lm_tp_truerad_supp_$l.py/g" "hd_lm_tp_truerad_$l.sub"; done
for l in {1..500}; do sbatch --time=168:00:00 --nodes=1 --ntasks=1 -A statdept hd_lm_tp_truerad_$l.sub; done
