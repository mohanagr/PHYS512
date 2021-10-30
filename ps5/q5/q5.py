import numpy as np
import matplotlib.pyplot as plt


def anfft(k, f):
	'''
	Function to obtain Fourier transform of a sine function of any frequency f
	'''
	
	a1 = (1-np.exp(-2J*np.pi*(k-f)))/(1 - np.exp(-2J*np.pi*(k-f)/N))
	a2 = (1-np.exp(-2J*np.pi*(k+f)))/(1 - np.exp(-2J*np.pi*(k+f)/N))
	return (a1-a2)/2J


if __name__ == '__main__':

	N = 50  # our number of samples or sampling frequency (since physical time is n/N seconds)
	n = np.arange(N) 
	f = 1.1 # this is our true frequency of the wave (no. of cycles per second)
	n = np.arange(N)
	y = np.sin(2*np.pi*f*n/N)
	truefft = np.fft.fft(y)

	k = n # k is also 0 to N-1
	anlfft = anfft(k,f)

	err = np.abs(anlfft-truefft)
	print("COMPARING DIFFERENCE BETWEEN ANALYTIC AND NUMPY")
	print(f"the maximum error is {err.max():4.2e} and the avg error is {err.mean():4.2e}") # very close to machine precision levels 1e-14


	# WINDOWING

	window = 0.5 - 0.5*np.cos(2*np.pi*n/N)
	wy = y*window
	wfft = np.fft.rfft(wy)
	# plt.plot(n,wy)
	plt.title('Reducing the leakage')
	plt.plot(np.abs(wfft),'r--*', label='windowed')
	plt.plot(np.abs(truefft)[:N//2],'.', label='not windowed')
	plt.text(5,10, "Windowed function has power in just 3 modes,\ncompared to almost 20 modes for non-windowed.")
	plt.legend()
	plt.savefig('./redcuing_leakage.png')
	plt.show()




