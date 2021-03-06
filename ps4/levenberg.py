#! /usr/bin/python3.6

import time
import numpy as np
import camb
from datetime import datetime


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


def deriv_TT(pars, lmax):
    if(pars[3]<0.01):
        raise Exception
    derivs = np.zeros((3049,6))
    delp = [0.001, 0.0001, 0.0001, 0.00001, 1e-11, 0.00001]   # deltas for each parameter
    pmat = np.tile(pars,len(pars)).reshape(len(pars),len(pars))
    pmat1 = pmat + delp*np.eye(len(pars))
    pmat2 = pmat - delp*np.eye(len(pars))
    # print(pars, "from get deriv TT")
    base = get_spectrum(pars)

    for i in range(len(pars)):
        derivs[:,i] = 0.5*(get_spectrum(pmat1[i,:])-get_spectrum(pmat2[i,:]))/delp[i]
    
    base = base[:lmax]
    derivs = derivs[:lmax,:]
    return base, derivs
    
def fit_lm(fun,m,lmax,y,N=None,niter=27,atol=1.e-2,rtol=1.e-12):
    print("init pars are:", m)
    def update_lambda(lamda,success):
        if success:
            lamda=lamda/1.5
            if lamda<0.5:
                lamda=0
        else:
            if lamda==0:
                lamda=1
            else:
                lamda=lamda*2
        return lamda

    lm = 1e4  # On some trial and error, I prefer starting from high LM, 
    #so that my initial step is not some random stupid Newton's step that throws me into an unknown space

    I = np.eye(len(m))
    chisqnew = 0
    if(N is None):
        N = np.eye(len(y))

    model,derivs=fun(m, lmax)
    r=y-model
    # print(model, y, N)
    Ninv = np.linalg.inv(N)
    # chisq= np.sum((r/np.diag(N))**2)
    chisq = r.T@Ninv@r
    for i in range(niter):
        
        lhs=(derivs.T@Ninv@derivs + lm*I) # first step is always Newton's
        rhs=derivs.T@Ninv@r
        dm = np.linalg.inv(lhs)@rhs
        m_trial = m + dm
        print('on iteration ',i,' chisq is ',chisq,' taking step ',m_trial, 'with lambda ', lm)
        try:
            model, derivs = fun(m_trial, lmax)
        except Exception as e:
            print("bad params ")
            lm = update_lambda(lm, False)
            continue
        r = y-model
        chisqnew = r.T@Ninv@r
        
        if(chisqnew<chisq):
            # accept the new step
            m = m_trial
            chisq = chisqnew
            lm = update_lambda(lm, True)
            print("step accepted. new m is", m)

            if((np.abs((chisqnew-chisq)/chisq)<rtol) and lm==0):
                # if lm=0, we're in Newton's domain, and fairly close to actual minima
                # Even if chain coverges before lm=0, let the temperature decrease and lm reach 0 before exiting
                param_cov = np.linalg.inv(derivs.T@Ninv@derivs)
                print("CHAIN CONVERGED")
                break
        else:
            # stay at the same point and try a more Gradient descent-ish step next
            lm = update_lambda(lm, False)
            if(lm>1e8):
                param_cov = np.linalg.inv(derivs.T@Ninv@derivs)
                print("CHAIN STUCK. TERMINATING")
                break

            print("step rejected. old m is", m)

    param_cov = np.linalg.inv(derivs.T@Ninv@derivs)
    np.savetxt(f'./param_cov_{datetime.now().strftime("%b%d_%H%M")}.txt', param_cov)
    return m

if __name__ == "__main__":

    dat = np.loadtxt("./COM_PowerSpect_CMB-TT-full_R3.01.txt", skiprows=1)
    y = dat[:,1]
    pars=np.asarray([65,0.02,0.1,0.07,2.00e-9,0.97]) # run 2 init params

 #  results
 #  [6.00080844e+01 2.12095628e-02 1.37395306e-01 1.00003801e-02, 1.99728939e-09 9.35198641e-01] run 1  -> ditching this. some weird, unphysical param space
 #  [6.81078776e+01 2.23451218e-02 1.17957964e-01 8.38390097e-02 2.21377445e-09 9.72410700e-01] run 2   -> this seems acceptable enough. Can use cov matrix

    err_y = 0.5*(dat[:,2] + dat[:,3])
    N = np.eye(len(y))*err_y**2
    # print(model,y,N )
    ells = dat[:,0]
    print(np.max(ells), ells.shape)

    newpars = fit_lm(deriv_TT, pars,len(y), y, N=N)
    print('best fit pars obtained are:\n', newpars)


