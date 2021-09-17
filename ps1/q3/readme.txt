In order to estimate the error of interpolation we shall use bootstrapping method in a way that is technically sound and provides stable results.

The method used here is functionally similar to cross-validation methods in machine learning, where the "true" test-error is unknown and needs to be estimated.

Like most real life problems, we do not know the "true" function underlying the Temperature-Voltage curve in the question. Therefore, it is not possible to generate an error estimate by comparison of interpolated value against any reference value. Therefore, we utilize the sampling variance aready present in the data as our starting point for error estimation.

We choose a random subset S of our data D, such that n(S) = f * n(D), where f in (0,1)  [n = number of points in set]. This is akin to resampling without replacement. Almost every resampled data set will be different from all the others.

The bootstrap method is based on the fact that these hundreds of resampled data sets comprise a good estimate of the sampling distribution for the derived/complicated quantity we're trying to compute. In the present problem, this quantity is the interpolated value of Temperature at a given test point x_test[i].

We can imagine it more physically as follows. 

For a test point, say 1.2 volts, where we'd like to estimate the value of temperature, we fit our interpolant 100 different times (using 100 different resamples). Since each subset of samples is different, each fit is also slightly different. Thus, we have 100 slightly different estimates of temperature at 1.2 volts. By the principle of bootstrapping, after a large number of such "simulations", we can expect that the variation of temperature values obtained resemble very closely the inherent (unknown) error distribution of the "true" T(V) [T as function of V]

Thus, we can write our error at point x_test[i] as stddev( y_ji ) [j = 1,n_simulations]   stddev of all y values for i_th input

Another, very useful inference that can be obtained from this method is that by varying the fraction "f" of original dataset that we use for resampling, we can cross-verify our error estimate.

Intuition tells us that if the number of samples reduce (only 10 voltage points vs 143 points) and we still interpolate over the whole range of Voltage values, the error should increase. There's a lot of missing information that we're trying to interpolate with very few data points. This is indeed the case, as shown in error_vs_samplesize.png

Basically, resampling with only 10 points = what error we'd have if we had 10 pts in original dataset. Since error increases with reducing the size of the sample, any fraction < 100% puts an upper bound on the "true" error with 143 points.