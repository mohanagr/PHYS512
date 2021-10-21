import time
import numpy as np
import camb
from scipy import stats

def get_spectrum(pars,lmax=3000):
    #print('pars are ',pars)
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    tau=pars[3]
    As=pars[4]
    ns=pars[5]
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[2:,0]   # monopole and dipole removed
    return tt

if __name__ == '__main__':
    
    
    planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
    ell=planck[:,0]
    y=planck[:,1]
    y_errs=0.5*(planck[:,2]+planck[:,3])
    
    print("H0 (null hypo):\tThe chisquare (summation of small errors) could've come from random fluctuations = Good model")
    print("H1 (alt  hypo):\tThe chisquare could not have come from random fluctuations = something bad in the model")
    print("\nLet us set confidence level at 99.9% i.e. alpha = 0.001. \nIf the chisquare is so large that the probability of it occuring randomly is <0.001, we can reject our Null Hypothesis.")
    
    pars=np.asarray([60,0.02,0.1,0.05,2.00e-9,1.0])
    y_pred=get_spectrum(pars)
    y_pred=y_pred[:len(y)]
    resid=y-y_pred
    chisq=np.sum( (resid/y_errs)**2)
    df = len(resid)-len(pars)

    print(f"\n The relevant critical value of chisquare is {stats.chi2.isf(0.001, df):6.2f}")
    print("So if our chisquare exceeds this value, we are 99% sure that our model is lacking in something and it's not a good fit.")

    
    
    print("\nFor Garbage params:")
    print(f"chisq is {chisq:6.2f} for {df} degrees of freedom.")
    print(f"DEFINITELY BAD")


    pars=np.asarray([69, 0.022, 0.12, 0.06, 2.1e-9, 0.95])
    y_pred=get_spectrum(pars)
    y_pred=y_pred[:len(y)]
    resid=y-y_pred
    chisq=np.sum( (resid/y_errs)**2)
    print("\nFor Better params:")
    print(f"chisq is {chisq:6.2f} for {df} degrees of freedom.")
    print(f"It's still bad. needs improvement. ")
    

