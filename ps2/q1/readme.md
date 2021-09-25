There's indeed a singularity encountered in custom integrator at z = R (where R is the radius of the shell). This is expected since 
electric field is a piece-wise continuous function, with discontinuity at z = R.

```
E = {
    
    0       ; z <  R
    1/z^2   ; z >= R
}
```

The warning obtained is:
```
q1.py:46: RuntimeWarning: invalid value encountered in true_divide
  return num/dem
```

Python basically skips the z = R point. In the graph, it can be seen that there's a small gap at z = 2 (= R)

The reason behind the small "kink" present near z ~ 1.8 to 2 is the fact that Legendre Polynomials are, in the end, polynomials. __And a polynomial can never be discontinuous__. Therefore, to model the sharp discontinuity, a polynomial has to "bend" or fall a little bit in order to rise sharply.