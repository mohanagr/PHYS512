import numpy as np # linear algebra
import pandas as pd
from scipy.interpolate import CubicSpline


# func : the function to sample and interpolate
# low: lower bound of input
# low: upper bound of input
# npts: number of points

def analyze_poly(func, low, high, npts):


    x_fit = np.linspace(low,high,npts)
    y_fit = func(x_fit)
    dx = x_fit[1] - x_fit[0]

    # let's generate some dummy input data where we'll test our interpolations
    rng = np.random.default_rng(seed=42)
    xmin = x_fit[1]
    xmax = x_fit[-3]
    x_test = np.sort(rng.random(100)*(xmax - xmin) + xmin)
    # x_test = np.linspace(xmin,xmax,1001)
    y_test = func(x_test)


    # First doing a cubic polynomial fit
    # odd order is preferred, so that all points are bracketed on both sides

    y_intrp_poly = np.nan*np.ones(x_test.shape)
    for i in range(len(x_test)):
        idx = (x_test[i]-x_fit[0])/dx
        idx = int(np.floor(idx))
    #     print(idx)
        # take 4 pts for cubic interpolation
        x_slice = x_fit[idx-1:idx+3]
        y_slice = y_fit[idx-1:idx+3]
        p = np.polyfit(x_slice,y_slice,3)
        y_intrp_poly[i] = np.polyval(p,x_test[i])

        
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(6,4)
    ax.set_title('actual data vs interpolated points -- poly fit')
    ax.plot(x_fit,y_fit, 'r.', label='actual data')
    ax.plot(x_test, y_intrp_poly,'g*', label='interpolated values at input')
    leg = ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.grid(True)
    # fig.savefig('./ps1_q3.png')
    MAE = np.mean(np.abs(y_test - y_intrp_poly))
    print(f"error in polynomial interpolation is {MAE:.2e}:")


def analyze_cs(func, low, high, npts):

    x_fit = np.linspace(low,high,npts)
    y_fit = func(x_fit)
    dx = x_fit[1] - x_fit[0]

    # let's generate some dummy input data where we'll test our interpolations
    rng = np.random.default_rng(seed=42)
    xmin = x_fit[1]
    xmax = x_fit[-3]
    x_test = np.sort(rng.random(100)*(xmax - xmin) + xmin)
    # x_test = np.linspace(xmin,xmax,1001)
    y_test = func(x_test)

    cs = CubicSpline(x_fit,y_fit)
    y_intrp_cs = cs(x_test)

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(6,4)
    ax.set_title('actual data vs interpolated points -- Cubic spline')
    ax.plot(x_fit,y_fit, 'r.', label='actual data')
    ax.plot(x_test, y_intrp_cs,'g*', label='interpolated values at input')
    leg = ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.grid(True)
    # fig.savefig('./ps1_q3.png')
    MAE = np.mean(np.abs(y_test - y_intrp_cs))
    print(f"error in polynomial interpolation is {MAE:.2e}:")


def rat_fit(x,y,n,m):
    assert(len(y)==n+m+1)
    assert(len(x)==len(y))
    # P = c0 + c1*x1...+ c_n*x_n
    # Q = 1 + d1*x1 ... + d_m*x_m
    
    mat = np.zeros([n+m+1, n+m+1])
    
    for i in range(n+1):
        mat[:,i] = x**i
    for i in range(1,m+1):
        mat[:,i+n] = -y*x**i

    #debug
    #print(np.linalg.eig(mat))

    coeff = np.dot(np.linalg.pinv(mat), y)   # pinv returns actual inv when available. will be very useful in Lorentzian interpolation
    
    c = coeff[:n+1]
    d = coeff[n+1:]
    
    return c, d

def rat_eval(c,d,x):

    P = 0
    Q = 1
    
    for i in range(len(c)):
        P = P + c[i]*x**i
    for i in range(len(d)):
        Q = Q + d[i]*x**(i+1)
        
    return P/Q

def analyze_ratfunc(func, low, high, npts):

    x_fit = np.linspace(low,high,npts)
    y_fit = func(x_fit)
    dx = x_fit[1] - x_fit[0]

    # let's generate some dummy input data where we'll test our interpolations
    rng = np.random.default_rng(seed=42)
    xmin = x_fit[1]
    xmax = x_fit[-3]
    x_test = np.sort(rng.random(100)*(xmax - xmin) + xmin)
    # x_test = np.linspace(xmin,xmax,1001)
    y_test = func(x_test)

    c, d = rat_fit(x_fit,y_fit,4,6)

    y_intrp_rat = rat_eval(c,d, x_test)

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(6,4)
    ax.set_title('actual data vs interpolated points -- Ratfunc')
    ax.plot(x_fit,y_fit, 'r.', label='actual data')
    ax.plot(x_test, y_intrp_rat,'g*', label='interpolated values at input')
    leg = ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.grid(True)
    # fig.savefig('./ps1_q3.png')
    MAE = np.mean(np.abs(y_test - y_intrp_rat))
    print(f"error in polynomial interpolation is {MAE:.2e}:")


if __name__ == '__main__':

    
    ###############################################################################################
    # all interpolations use same points (number and location) for estimation for fair comparison #
    ###############################################################################################

    # first let's compare errors for cos(x)
    func = np.cos

    low = -np.pi/2
    high = np.pi/2
    analyze_poly(func, low, high, 11) # accuracy ~ 1.e-4
    analyze_cs(func, low, high, 11) # accuracy ~ 1.e-5
    analyze_ratfunc(func,low, high, 11) # accuracy ~ 1.e-10

    # now let's switch to Lorentzian

    func = lambda x: 1/(1+x**2)

    low = -1
    high = 1

    analyze_poly(func, low, high, 11) # accuracy ~ 1.e-4
    analyze_cs(func, low, high, 11) # accuracy ~ 1.e-5

    ### NOTE ###
    # on using linalg.inv .. the interpolations are grossly wrong, as shown by the graph
    # accuracy is totally meaningless
    # 
    # on debugging the eigenvalues of this function (uncomment the print statement in ratfit)...
    # ...it is seen that some eigenvalues are extremely small < 1.e-17. They cause things to blow up
    # below function call uses pinv

    analyze_ratfunc(func,low, high, 11) # accuracy ~ 1.e-16 














