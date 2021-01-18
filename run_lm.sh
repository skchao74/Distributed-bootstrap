#!/usr/bin/env bash

for l in {1..1000}; do mkdir $l; done
for l in {1..1000}; do cp cv_hd_lm_tp.py cv_hd_lm_tp_$l.py; sed -i "s/l = 1/l = $l/g" "cv_hd_lm_tp_$l.py"; done
for l in {1..1000}; do cp cv_hd_lm_tp.sub cv_hd_lm_tp_$l.sub; sed -i "s/cv_hd_lm_tp.py/cv_hd_lm_tp_$l.py/g" "cv_hd_lm_tp_$l.sub"; done
for l in {1..1000}; do sbatch --time=168:00:00 --nodes=1 --ntasks=1 -A statdept cv_hd_lm_tp_$l.sub; done
