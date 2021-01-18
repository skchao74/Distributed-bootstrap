### Reproduce the code

Here, we take linear model with Toeplitz design (in lm_tp folder) as an example. Linear model with equi-correlation design, GLM with Toeplitz design, and GLM with equi-correlation design are similar to this.

1) cv_hd_lm_tp.py contains the base program for running one replication of bootstrap CI algorithms. cv_hd_lm_tp.sub contains the scripts for loading the required module and running the .py file.
2) run_lm_tp.sh generates .py and .sub files for 1000 replications of bootstrap CI algorithms and submits the jobs to computing clusters.
3) postprocess_lm_tp.sh combines the output .csv files across 1000 replications.
4) hd_lm_tp_truerad_supp.py contains the base program for running one replication for simulating oracle widths. hd_lm_tp_truerad_supp.sub contains the scripts for loading the required module and running the .py file.
5) run_lm_tp.sh generates .py and .sub files for 500 replications for simulating oracle widths and submits the jobs to computing clusters. The results will be saved in .csv files.
6) postprocess_lm_tp_truerad.sh combines the output .csv files across 500 replications.
7) plots_lm.py aggregates the results to compute empirical coverage probabilities, average widths, and oracle widths, and generates the plots.
