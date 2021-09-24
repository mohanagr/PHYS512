import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import quad

def integrateLegendre(func,npts,a, b, *args, **kwargs):
    
    # general purpose legendre integrator

    # we first get coefficients for a n-1 order legendre function that goes through n points (for a square Vandermonde matrix)
    # then we estimate the integral as 2*coefficient of P_0

    # x is rescaled to -1, 1
    x = np.linspace(-1, 1, npts)
    m = (b-a)/2
    c = (b+a)/2
    Y = func(m*x+c, *args, **kwargs) # equivalenty we could pass an xnew = np.linspace(a, b, npts) and avoid m and c
    P = np.polynomial.legendre.legvander(x,npts-1)
    Pinv = np.linalg.pinv(P)
    coeff = Pinv@Y
#     print(coeff.shape)
    integral = 2*coeff[0]
    integral = integral * (b-a)/2  
    
    return integral

def d_E(theta, z, R):
    """Integrand for obtaining Electric field at a distance z from the center of a thin spherical shell of radius R

    Parameters
    ----------
    theta: float
        Variable to integrate over using any integration routine
    z: float
        Distance from the center of the shell where Electric field needs to be evaluated
    R: float
        Radius of the shell

    Returns
    -------
    float
        dE - infinitesimal electric field
    """

    num = (z - R*np.cos(theta)) * np.sin(theta)
    dem = (z**2 + R**2 - 2*R*z*np.cos(theta))**(3/2)
    
    return num/dem
    
if __name__ == '__main__':
    
    R = 2
    z = np.linspace(1,3,900)
    z = np.sort(np.append(z,2))
    # y_exp = integrateLegendre(myfunc, 5, 0, np.pi)

    E_vals = [integrateLegendre(d_E, 51, 0, np.pi, z_i, R) for z_i in z]

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(6,4)
    ax.set_title('Legendre integrator, Radius=2')
    ax.plot(z, E_vals,'r-', label="Electric field")
    leg = ax.legend()
    ax.set_xlabel('z ')
    ax.set_ylabel('E (arbitrary units)')
    ax.grid(True)
    fig.savefig('./ps2_q1_ElectricField_myIntegration.png')


    # trying scipy quad

    E_vals = [quad(d_E,0, np.pi, args=(z_i, R))[0] for z_i in z]

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(6,4)
    ax.set_title('Scipy Quad, Radius=2')
    ax.plot(z, E_vals,'r-', label="Electric field")
    leg = ax.legend()
    ax.set_xlabel('z')
    ax.set_ylabel('E (arbitrary units)')
    ax.grid(True)
    fig.savefig('./ps2_q1_ElectricField_SciPyQuad.png')
