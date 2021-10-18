## File structure

The code to run an MCMC chain is present in `mcmc.py`.
The code to run Levenberg-Marquardt minimization is present in 'levenberg.py'
The analysis and visualization results of MCMC chains (for questions 3 and 4) is present in 'MCMC_analysis.ipynb'. 
(It can be viewed as in on Github, but some LateX equations may not be displayed correctly depending on your browser. Running it locally is recommended.)

## Output structure

Two L-M chains were run with, and the parameter covariances obtained from those are present in files `param_cov_Oct*.txt`. The latest of these outputs (Oct13_1322) was used to initiate the MCMC chain for Q3.

The best-fit parameters obtained in this L-M run were: 

`[6.81078776e+01 2.23451218e-02 1.17957964e-01 8.38390097e-02 2.21377445e-09 9.72410700e-01]`

Even though the code prints these parameters, the output has not been recorded separately since it is the covariance matrix we are most interested in. We know that L-M best-fit parameters will not be accurate in 20-30 steps.

