import matplotlib.pyplot as plt
import numpy as np

def exp(x, a=1):
    
    # normalized on 0 to inf

    return np.exp(-x)

def lorentz(x,a=1):
    
    # unnormalized
    # we can scale the range later

    return a/(1+x**2)

def gauss(x,a=1):

    return a*np.exp(-0.5*x**2)

if __name__ == '__main__':
	
	# compare different functions:

	# x = np.linspace(0,10,1000)
	# plt.plot(x,exp(x), label='exponential')
	# plt.plot(x,lorentz(x), label='lorentzian')
	# plt.plot(x,gauss(x), label='gaussian')
	# plt.legend()
	# plt.savefig('./func_comparison.png')

	# plt.clf()

	# # as apparent from figure we need to scale lorentz very slightly 
	# # so that at x=0 it doesnt touch the exp()

	size = 10000000

	a = 1.05 # scaling the lorentz up
	rands = a*np.pi/2*np.random.rand(size) # y = tan(x/a), and y needs to go from 0 to inf.
	x_env = np.tan(rands/a)
	r = exp(x_env)/lorentz(x_env,a)
	decision = np.random.rand(size)
	accept = decision<r

	x_accepted = x_env[accept]

	bin_vals, bin_edges = np.histogram(x_accepted,bins=100,density=False)
	bin_centers = 0.5*(bin_edges[:-1]+bin_edges[1:])
	width = bin_centers[1]-bin_centers[0]
	y = exp(bin_centers)
	plt.bar(bin_centers, bin_vals/bin_vals.sum(), width=width, edgecolor='yellow',label='sampled')
	plt.plot(bin_centers, y/y.sum(),linewidth=2,c='r',label='PDF')
	plt.xlim([0,5]) # we dont want to view all the high random numbers we might have gotten
	plt.legend()
	plt.text(2,0.08, f"acceptance rate: {np.mean(accept)*100:2.0f}%")
	plt.savefig('./output_lorentz.png')


	# Now let's try box method where we remap our y range from inf to finite

	# use mapping:
	# y = arctan(x) so that f(x) goes to g(y)

	# see how the new PDF g(y) looks like
	y=np.linspace(0,np.pi/2,1001)
	cents=(y[1:]+y[:-1])/2

	#prob = exp(-x) goes to exp(-(tan(y))/cos^2(y)
	pp=np.exp(-np.tan(cents))/np.cos(cents)**2
	plt.clf()
	plt.plot(cents,pp)
	plt.title('Exponential PDF remapped using y = arctan(x)')
	plt.savefig('./remapped_PDF.png')

	# since it goes to zero, we can just sample in a box now

	z=np.pi*np.random.rand(size)/2
	h=np.random.rand(size)*1.05 # slightly bigger box that function at x=0
	accept=h<np.exp(-np.tan(z))/np.cos(z)**2
	z = z[accept]

	# convert the new variable back to original
	y = np.tan(z)
	bin_vals, bin_edges = np.histogram(y,bins=100,density=False)
	bin_centers = 0.5*(bin_edges[:-1]+bin_edges[1:])
	width = bin_centers[1]-bin_centers[0]
	f = np.exp(-bin_centers)
	plt.clf()
	plt.bar(bin_centers, bin_vals/bin_vals.sum(), width=width, edgecolor='yellow',label='sampled')
	plt.plot(bin_centers, f/f.sum(),linewidth=2,c='r',label='PDF')
	plt.text(2,0.08, f"acceptance rate: {np.mean(accept)*100:2.0f}%")
	plt.xlim([0,5])
	plt.legend()
	plt.savefig('./output_box.png')
