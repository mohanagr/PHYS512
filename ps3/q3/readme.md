## Noise model

If there are two models with differening chisquare values, we prefer the one with lesser chisquare -- as it's more probable that the data we see came from the model with lesser chisquare.

`P(d|m1) > P(d|m2)` if `chisq(m1) < chisq(m2)`

When starting with the default model of Identity Noise matrix, (which translates to assuming that the error in our estimates are distributed as standard normal `N(0,1)`), we obtain a chisquare of 6745. In search for a new noise model, we first plot the residuals with the values of the dependent variable (called `y` in the code and `d` in Jon's notes). This is to visualize if there is variation in the magnitude of error in our predictions -- small for some points, larger for others.

It is apparent from `residual_variation.png` that there is a _heteroskedastic_ trend in our residuals, i.e. the variation of residuals is: 
    1. Not constant
    2. Not unity

The simplest update to our noise matrix would be to replace `N(i,i) = [residual(i)]**2`. However, this scheme is susceptible to failure if residual for some of our points is ~ 0. These quantities would blow up in `inv(N)`, and would bias our model too much. We could get a slightly better solution if our points could "communicate" with their neighbours. This would ensure that no one point has disproportionate effect on the model.

A way to achieve such a model is to bin the data and assign the average error in each bin as the error for those points in the nosie matrix. There are two ways to estimate the said "error". We could either:

- Calculate the variance of residuals in the bin.

- Take the mean of the square of the residuals.

In practice, both of the above ways yield similar answers because `Var(e) = <e^2> - <e>^2`, and the second term on RHS (the mean of residuals) in zero or close to zero in most cases (including ours).

The data has been segregated into ten equally spaced bins for the noise estimate used in this code.

## Result

With the new noise matrix, our chisquare reduces to 441 -- improvement of a factor of 15! This model can now be used to estimate model parameters and uncertainties.

## Bonus

There are two parts to the bonus question.

1. Introducing non-circularity in the paraboloid.

    In this case, the coefficients of `(x-x0)^2` and `(y-y0)^2` will be different, unlike in the previous case. If we look down on the paraboloid (facing the bottom), we'll see an ellipse rather than a circle.

2. Introducing asymmetric axes.

    In this case, the paraboloid's principal axes are rotated with respect to ground. A simple coordinate transformation using 2-D rotation matrix will yield the correct coordinates.

First of the above two cases has been implemented in the code. Second part, though straightforward, has been skipped because of time-constraints.

The paraboloid is very close to a perfect circle, and one of the axes is 0.35% larger than the other.

