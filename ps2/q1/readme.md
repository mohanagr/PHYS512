There's indeed a singularity encountered in custom integrator at z = R (on the surface of the shell).

The warning obtained is:
```
q1.py:46: RuntimeWarning: invalid value encountered in true_divide
  return num/dem
```

Python basically skips that point. In the graph, it can be seen that there's a small gap at z = 2 (= R)