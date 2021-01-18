#!/usr/bin/env bash

for a in k nk1; do
for b in p95 p9; do
for c in cov rad; do
for d in inf 2 1 pt; do
for e in t1 t2 t3 t4; do
ls */hd_lm_tp_supp_${a}_${b}_${c}_${d}_${e}.csv | xargs cat > hd_lm_tp_supp_${a}_${b}_${c}_${d}_${e}.csv
done
done
done
done
done