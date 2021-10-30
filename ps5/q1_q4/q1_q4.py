import numpy as np
import matplotlib.pyplot as plt

def conv1d(x, kernel):

    '''
    One dimensional "linear" convolution. Exact output as that of numpy.convolve(mode='full')
    This gives "wrap-around safe" output
    total length of output = N + M -1 (N = len(x), M = len(kernel))
    '''
    
    nk = len(kernel)
    nx = len(x)
    xnew = np.zeros(nx+2*nk-2)
    conv = np.zeros(nx+nk-1)
    xnew[nk-1:nk+nx-1] = x
    
    for i in range(len(xnew)-nk+1):
        s = 0
        for j in range(nk):
            s += kernel[nk-j-1] * xnew[i+j]
            
        conv[i] = s
    return conv

def fftconv(x, y):
    '''Take convolution of two arrays using Fourier transforms. The magnitude is not normalized (factors of 1/N)'''

    return np.real(np.fft.ifft(np.fft.fft(x)*np.fft.fft(y)))

def corr1d(x, kernel):

    '''
    One dimensional linear correlation. Exact output as that of numpy.correlate(mode='full')
    This gives "wrap-around safe" output
    '''
    
    nk = len(kernel)
    nx = len(x)
    xnew = np.zeros(nx+2*nk-2)
    corr = np.zeros(nx+nk-1)
    xnew[nk-1:nk+nx-1] = x
    
    for i in range(len(xnew)-nk+1):
        s = 0
        for j in range(nk):
            s += kernel[j] * xnew[i+j]
            
        corr[i] = s
    return corr

def circonv1d(x, kernel):
    '''
    One dimensional "circular" convolution. This wraps around the output, since that is how DFTs work.
    By definition the length of x and kernel should be the same (since FT(x) x FT(kernel) won't work if dimesnions not same).
    This is fully equivalent to taking fourier transforms, multiplying and then taking IFT. This is the hard way.
    '''
    assert(len(x)==len(kernel))
    N = len(x)
    conv = np.zeros(N)
    for i in range(N):
        s = 0
        for j in range(N):
            s += x[j]*kernel[i-j]
        conv[i]=s
    return conv

def fftcorr(x,y):
    assert(len(x)==len(y))

    '''Take correlation of two function using Fourier transforms. The magnitude is not normalized (factors of 1/N)'''

    return np.fft.ifft(np.fft.fft(x)*np.conjugate(np.fft.fft(y)))

def array_shift(x, m):
    '''
    Shift array x by m. m can be positive or negative. m can also be greater than length of x, since shift is circular
    '''
    kernel = np.zeros(len(x))
    kernel[m%len(x)] = 1
    return circonv1d(x,kernel)



if __name__ == '__main__':
        
    # Q1 - shift a gaussian
    x = np.linspace(0,10,101)
    gauss = lambda x: np.exp(-0.5*(x-5)**2) # a gaussian that starts in the center of the array
    y = gauss(x)
    #let's shift it by half the array length
    shifted_y = array_shift(y,len(x)//2)
    # plt.plot(y, label='original, peak at index 51')
    # plt.plot(shifted_y, label='shifted by 50')
    # plt.xlabel('indices')
    # plt.ylabel('value')
    # plt.xticks(np.arange(0,101,10))
    # plt.legend(loc=1)
    # plt.savefig('./gauss_shift.png')
    # plt.clf()

    # Q2 - correlation of gaussian with itself
    # corr = fftcorr(y,y)
    # plt.title('Correlation of Gaussian with itself')
    # plt.plot(np.abs(corr))
    # plt.xlabel('shift amount')
    # plt.ylabel('value (arbitrary units)')
    # plt.savefig('./gauss_corr_itself.png')
    # plt.clf()

    #Q3 correlation of a gaussian with a shifted gaussian
    # we should expect a peak at 50 if we correlate the above two gaussians, since we shifted it by 50

    # corr = fftcorr(shifted_y, y)
    # plt.title('Correlation of Gaussian with shifted version')
    # plt.plot(np.abs(corr))
    # plt.xlabel('shift amount')
    # plt.ylabel('value (arbitrary units)')
    # plt.xticks(np.arange(0,101,10))
    # plt.axvline(50, c='r', linestyle='--')
    # plt.savefig('./gauss_corr_shifted.png')
    # plt.clf()

    #Q4 "wrap around safe" convolution. This is generally referred to as "linear" convolution in many books

    x = [1,2,3,4]
    k = [0,1,0,0]
    circ = circonv1d(x,k)
    ft = fftconv(x,k)
    lin = conv1d(x,k)
    print("array is:\t", x)
    print("kernel is:\t", k)
    print("DFT conv is:\t", ft)
    print("circular manual conv:\t", circ)
    print("linear conv is:\t", lin)
    print("We expected N+M-1 = 7 elements in our full linear convolution.")
    print("\nThis method allows shapes of kernel and array to be different.\nE.g. Let's keep kernel [0,1] to shift elements by 1\n")
    x = [1,2,3,4]
    k = [0,1]
    lin = conv1d(x,k)
    print("array is:\t", x)
    print("kernel is:\t", k)
    print("linear conv is:\t", lin)

    print("\nExample: let's calculate 1-D Gravitational potential\n")

    # point charges
    x = [1,1,1,1]
    # distance kernel
    k = [100000,1,0.5,0.33,0.25,0.125,0.0625,0]
    #potential
    lin = conv1d(x,k)
    print("array is:\t", x)
    print("This is array of unit charges")
    print("kernel is:\t", k)
    print("this is a kernel of 1/r")
    print("linear conv is:\n", lin)
    print("this is how potential varies on our 1-D grid because of 4 charges.")


