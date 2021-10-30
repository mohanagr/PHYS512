import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
	
	# let us generate a bunch of random walks so that we can take an ensemble average
	# (for a single RW, power will fluctuate, but in an averaged power spectrum we can clearly see 1/f**2 dependence)
	nsteps = 10000
	nwalks = 1000
	rwalks = np.zeros((nwalks,nsteps))
	walkfts = np.zeros((nwalks, nsteps), dtype='complex')
	for i in range(nwalks):
		rwalks[i,:] = np.cumsum(np.random.randn(nsteps))
		walkfts[i,:] = np.fft.fft(rwalks[i,:])
		
		#uncomment below lines to plot a randomwalk. done once and saved.
	# 	plt.plot(rwalks[i,:])

	# plt.title('Gaussian random walk visualization')
	# plt.xlabel('time')
	# plt.ylabel('position')
	# plt.savefig('./random_walk_viz.png')
	# plt.clf()

	avg_power = np.mean(np.abs(walkfts)**2/nsteps**2, axis=0)
	c = avg_power[1] # get the normalizing factor of numerator from first term to plot theoretical prediction of 1/k**2
	# best fit may give slightly better result but that's overkill here.

	k = np.arange(1,20)
	theory = c/k**2
	print(avg_power.shape)
	plt.title('Power spectrum of a Gaussian random walk')
	plt.plot(k,avg_power[1:20],'r*',label='obtained')
	plt.plot(k,theory, 'b.', label='1/k**2')
	plt.xticks(k)
	plt.legend()
	plt.xlabel('Fourier mode')
	plt.ylabel('Power')
	plt.savefig('./ps_randomwalk.png')
	plt.show()