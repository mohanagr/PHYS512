import matplotlib.pyplot as plt
import numpy as np


n = 1000000
u1 = np.random.rand(n)
u2 = np.random.rand(n)*0.73576
x = u2/u1
accept = (u1**2 <= np.exp(-x))
vals_x = x[accept]
vals_u1 = u1[accept]
vals_u2 = u2[accept]
plt.title('Acceptance area')
plt.plot(vals_u1, vals_u2, '.')
plt.xlabel("U1")
plt.ylabel("U2")
plt.savefig('./acceptance_area.png')
plt.clf()

bin_vals, bin_edges = np.histogram(vals_x,bins=100,density=False)
bin_centers = 0.5*(bin_edges[:-1]+bin_edges[1:])
width = bin_centers[1]-bin_centers[0]
f = np.exp(-bin_centers)
plt.bar(bin_centers, bin_vals/bin_vals.sum(), width=width, edgecolor='yellow',label='sampled')
plt.plot(bin_centers, f/f.sum(),linewidth=2,c='r',label='PDF')
plt.xlim([0,5])
plt.xlabel("x bins")
plt.ylabel("prob(x)")
plt.legend()
plt.savefig('output_uniform_ratios.png')
