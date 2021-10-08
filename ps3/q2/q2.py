import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def derivs(t, n, hlf):

    # A general first-order decay rate-equation is given as:
    # dn/dt  = -k * n

    k = np.log(2)/hlf  # rate constants
    dndt = np.zeros(len(hlf)+1)
    dndt[:-1] = -k * n[:-1]
    dndt[1:] = dndt[1:] + k*n[:-1]

    return dndt

if __name__ == '__main__':
    
    # Half lives of decay products in years
    hl = np.array([
    4.468e9,     #U-238
    0.066,       #Th-234
    0.00076484,  #Pa-234
    245500,      #U-234
    75380,       #Th-230
    1600,        #Ra-226
    0.010475,    #Rn-222
    5.898e-6,    #Po-218
    5.0989e-5,   #Pb-214
    3.78615e-6,  #Bi-214
    5.21e-12,    #Po-214
    22.3,        #Pb-210
    5.015,       #Bi-210
    0.3791,      #Po-210
    #Pb-206 stable
    ])


    n0 = np.zeros(len(hl)+1)
    n0[0] = 10000 #10k U-238 atoms, can be an arbitrary number
    tstart=0
    tend=1e10 # equivalent to 10 half-lives of U-238 

    #create logarithmically spaced points, to cover the whole 10^10 range in few points
    teval = np.linspace(0,np.log10(tend),1000)
    teval = 10**(teval)
    # teval = np.linspace(0,10,1000)

    # Use an implicit method like Radau as this is a stiff system of equations.
    N_out = solve_ivp(derivs,[tstart,tend],n0,method='Radau',t_eval=teval,args=(hl,)).y

    #Analyticaly we expect N_Pb206/N_U238 to be:
    k_U238 = np.log(2)/hl[0]
    Pb_U_theory = (np.exp(k_U238*teval)-1)

    fig, ax = plt.subplots(1,2)
    fig.set_size_inches(18,8)

    ax[0].set_title('Pb206/U238 over full range - half-log plot')
    ax[0].plot(teval, Pb_U_theory,'g-', label="Theoretical abundance ratio")
    ax[0].plot(teval, N_out[-1,:]/N_out[0,:],'r',linestyle=(0, (5, 10)), lw=3,label="Obtained abundance ratio")
    leg = ax[0].legend()
    ax[0].set_xscale('log')
    ax[0].set_xlabel('Time (years)')
    ax[0].set_ylabel('Ratio')
    ax[0].grid(True)

    ax[1].set_title('Pb206/U238 theory comparison - linear plot')
    ax[1].plot(teval, Pb_U_theory,'g-', label="Theoretical abundance ratio")
    ax[1].plot(teval, N_out[-1,:]/N_out[0,:],'r', linestyle=(0, (5, 10)), lw=3,label="Obtained abundance ratio")
    leg = ax[1].legend()
    ax[1].set_xlabel('Time (years)')
    ax[1].set_ylabel('Ratio')
    ax[1].grid(True)
    fig.savefig('./Pb206_U238.png')

    fig, ax = plt.subplots(1,2)
    fig.set_size_inches(18,8)
    ax[0].set_title('Th230/U234 over full range - half-log plot')
    ax[0].plot(teval, N_out[4,:]/N_out[3,:],'r-', label="Obtained abundance ratio")
    leg = ax[0].legend()
    ax[0].set_xscale('log')
    ax[0].set_xlabel('Time (years)')
    ax[0].set_ylabel('Ratio')
    ax[0].grid(True)

    ax[1].set_title('Th230/U234 over interesting range - linear plot')
    ax[1].plot(teval[0:650], N_out[4,0:650]/N_out[3,0:650],'b-', label="Obtained abundance ratio")
    ax[1].axvline(x=1.5e6,ymin=0,ymax=1,c='black',linestyle='--')
    ax[1].text( 1.6e6,0.2,"Long-term average reached after\n1.5 mil years.")
    leg = ax[0].legend()
    ax[1].set_xlabel('Time (years)')
    ax[1].set_ylabel('Ratio')
    ax[1].grid(True)

    fig.savefig('./Th230_U234.png')
