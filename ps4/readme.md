## File structure

The code for question one is present in `q1.py`.

The code to run an MCMC chain is present in `mcmc.py`.

The code to run Levenberg-Marquardt minimization is present in `levenberg.py`

The analysis and visualization results of MCMC chains (for questions 3 and 4) is present in `MCMC_analysis.ipynb`. *A detailed note on Importance Sampling is provided in it.*

(It can be viewed on Github, but some LateX equations may not be displayed correctly depending on your browser. Running it locally is recommended.)

## Output structure

Output for Q1 is present in `output_q1.txt`

Two L-M chains were run and the parameter covariances obtained from those are present in files `param_cov_Oct*.txt`. The latest of these outputs (`Oct13_1322.txt`) was used to initiate the MCMC chain for Q3.

The best-fit parameters obtained in this L-M run were: 

`[0.68 0.0223 0.117 0.084 2.2e-09 0.972]`. These are respectively: _H0_, _Ohmbh2_, _Ohmch2_, _Tau_, _As_, and _ns_.

Even though the code prints these parameters, the output has not been recorded separately since it is the covariance matrix that we are most interested in. We know that L-M best-fit parameters will not be accurate in 20-30 steps.

The parameter covariance matrix estimated using Importance Sampling of the Q3 chain is present as `param_cov_tau.txt`. I calculate the variances of parameters in two different ways: manual weighting, and `numpy.cov()`. It is shown in the analysis ipython notebook that both methods give the same results.

## Quality of Results

For Question 1, I use the chi-squared goodness-of-fit test for Maximum Likelihood Estimates (as described in Numerical Recipes). The critical value of chi-square at 99.9% confidence level is 2725. It is shown in `output_q1.txt` that the chisquare obtained using parameters suggested in Q1 give us a much higher chi-square. Therefore, the chi-square has certainly not come from random fluctuations (as we expect it to, since it's just a sum of randomly distributed errors), and improvements in model are needed.

Here it must be mentioned that even if the obtained chi-square is less than the variance of chi-square distribution with 2501 degress of freedom (~roughly 5000), it does not mean that the fit is good. Chi-square distribution with high degrees of freedom is heavily peaked, and does not follow the Gaussian rule of 1-sigma ~ 68%. A goodness-of-fit test such as above is necessary.

For all other questions, the chisquare is sufficiently below 2725.

The parameter constrains and 1-sigma errors are in surprisingly good agreement with Planck 2018 results _(Planck Collaboration 2018 results Part VI Cosmological Parameters)_ as shown in Jupyter notebook. This is especially the case after including the Tau prior, but even with uniform priors (Q3 run), the values for _H0_, _Ohmbh2_, _Ohmch2_ match the Planck results to at least 2 significant digits.

The constraint on Tau obtained from importance sampling and the one obtained from running a new chain with prior match to 2 siginificant digits (a difference of merely 0.001).

Parameter distributions have been visualized using `Seaborn` library (specifically the `pairplot` feature) instead of `Corner` plots, since I feel the plots with Seaborn are more visually appealing. I'm particularly satisfied with the plots obtained using final chain with 10K points for Q4.

