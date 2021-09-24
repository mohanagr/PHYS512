There's indeed a singularity encountered in custom integrator at z = R (on the surface of the shell).

The warning obtained is:
```
q1.py:46: RuntimeWarning: invalid value encountered in true_divide
  return num/dem
```

Python basically skips that point. In the graph, it can be seen that there's a small gap at z = 2 (= R)

The reason behind the small "kink" present near z ~ 1.8 to 2 is the fact that Legendre Polynomials are, in the end, polynomials. __And a polynomial can never be discontinuous__. Therefore, to model the sharp discontinuity, a polynomial has to "bend" or fall a little bit in order to rise sharply.