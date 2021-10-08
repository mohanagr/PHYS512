## Number of function evaluations

As explained in the attached pdf, a normal RK4 calls the derivative function 4 times per step. Whereas the augmented "double-stepper" calls it 11 times (4 times for single step of h, and 7 times for two steps of h/2) per step. Therefore, if number of function calls are to be the same in both functions, following relation has to be followed:

`4 * nsteps_RK4 = 11 * nsteps_RK4db`

If `nsteps_RK4` is 200, then `nsteps_RK4db` is roughly 73.

Using the double-stepper, we can sneak one more order of accuracy and make our estimate 5th order accurate (leading order error-term of sixth order in h). This is clearly apparent from the output. The new function is 10 times more accurate than the older-one **for the same amount of computational work**.

### Side-note

The small kink seen in error graph for RK4-double-steps is because of fairly large step-size (just 73 steps). It disappears when more steps are added, and a smooth error-curve is obtained.