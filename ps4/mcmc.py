import multiprocessing as mp
import time
import numpy as np
import camb
from datetime import datetime

def get_spectrum(pars,lmax=3000):
    #print('pars are ',pars)
    if(pars[3]<0.):
        raise Exception
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


def mcmc(params, param_cov, func, y, Noise=None, niter=1000):
    '''
        customized for get_spectrum() so x has been skipped from arglist. we don't need x.
        by default writes chain file to current directory
    '''

    if(Noise is None):
        Noise = np.eye(len(y))
    Ninv = np.linalg.inv(Noise)
    maxell = y.shape[0]
    print(f"Init params are: {params}")
    # take first step
    st = time.time()
    ynew = get_spectrum(params)
    ynew = ynew[:maxell]
    r = y-ynew
    chisq = r.T@Ninv@r
    et = time.time()
    print(f"Time per step: {et-st:4.2f}s")

    npar = len(params)
    
    chain = np.zeros((niter, npar+2)) # one col for chisq
    print(chain.shape)

    i=0
    while(True):
        if(i>niter):
            break
        trial_params = np.random.multivariate_normal(params, param_cov)
        try:
            ynew = get_spectrum(trial_params)
        except Exception as e:
            print("bad params. trying again")
            continue
        
        ynew = ynew[:maxell]
        r = y-ynew
        chisq_trial = r.T@Ninv@r
        # post_trial = np.exp()
        accept_prob = np.exp(0.5*(chisq-chisq_trial))
        if(np.random.rand(1)<accept_prob):
            params = trial_params
            chisq = chisq_trial
        
        chain[i,0]=i
        chain[i,1]=chisq
        chain[i,2:] = params
        print(f'On step: {i:d} chisq: {chain[i,1]:6.3f} H0: {chain[i,2]:4.2f} Ohmbh2: {chain[i,3]:6.4f} Ohmch2: {chain[i,4]:6.4f} Tau: {chain[i,5]:5.3e} As: {chain[i,6]:5.3e} ns: {chain[i,7]:6.4f}')
        i = i+1

    np.savetxt(f'./chain_{datetime.now().strftime("%b%d_%H%M")}.txt', chain)
    return

if __name__ == '__main__':
    
    param_cov = np.loadtxt('./param_cov_Oct13_1322.txt')

    # params_init = np.asarray([65,0.02,0.1,0.07,2.00e-9,0.97])
    params_init = np.asarray([64.3  ,0.02,0.1,0.054,2.00e-9,0.98])
    dat = np.loadtxt("./COM_PowerSpect_CMB-TT-full_R3.01.txt", skiprows=1)
    y = dat[:,1]
    err_y = 0.5*(dat[:,2] + dat[:,3])
    N = np.eye(len(y))*err_y**2

    mcmc(params_init, param_cov, get_spectrum, y, Noise=N, niter=8000)


