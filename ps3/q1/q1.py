import numpy as np
import matplotlib.pyplot as plt

def rk4_step(func, x, y, h):
    
    k1 = h*func(x,y) #get the sloped at initial point
    k2 = h*func(x+h/2, y+k1/2) # use the slope to get to the midpoint, and get a new slope
    k3 = h*func(x+h/2, y+k2/2) # use the slope to reevaluate the midpoint, and get a new slope
    k4 = h*func(x+h, y+k3) # use the slope to evaluate y at endpoint and get a new slope at endpoint
    ynew = y + k1/6 + k2/3 + k3/3 + k4/6
    return ynew

def rk4_stepd(func, x, y, h):

    
    #### for stepsize h ####
        
    H = h
    
    k1 = H*func(x,y)              # first function evaluation will be common for both h and h/2
    k2 = H*func(x+H/2, y+k1/2)
    k3 = H*func(x+H/2, y+k2/2)
    k4 = H*func(x+H, y+k3)
    ynew1 = y + k1/6 + k2/3 + k3/3 + k4/6
    
    #### for two steps of stepsize h/2 ####
    H = h/2
    
    k1 = k1/2 #reusing the previous k1 to save 1 evaluation
    k2 = H*func(x+H/2, y+k1/2)
    k3 = H*func(x+H/2, y+k2/2)
    k4 = H*func(x+H, y+k3)
    ymid = y + k1/6 + k2/3 + k3/3 + k4/6     # we have just gotten to ymid = y(x+h/2). gotta repeat to get to y(x+h)
    
    x = x + h/2
    y = ymid
    k1 = H*func(x,y)
    k2 = H*func(x+H/2, y+k1/2)
    k3 = H*func(x+H/2, y+k2/2)
    k4 = H*func(x+H, y+k3)
    ynew2 = y + k1/6 + k2/3 + k3/3 + k4/6
    
    ####--------- -------imp note--------------------- ####
    # ynew2-ynew1 ~ O(h^5) ~ error estimate of RK4 method #
    # can be used to make step size adaptive              #
    #-----------------------------------------------------#
    
    # as shown in Github readme, we can sneak one more order of accuracy by being cunning
    yfin = (16*ynew2-ynew1)/15
    
    return yfin

def solve(func, xlim, nsteps, y0, stepper):
    """Generic differential equation solver using a given function to calculate next step (like RK4)


    Parameters
    ----------
    func : function
        python function to integrate. Should return an array of dy/dx for each y[i]
    y0: float
        Initial value of y at x[0]
    nsteps: int
        Number of steps to integrate over
    xlim: float
        array [x_start, x_end]
    stepper: function
        Python function that calculates y(x+h) given x, y(x), dy/dx

    Returns
    -------
    array of floats
        y[i] for each x[i]
    """

    # x_arr[0] should be x0

    x_arr = np.linspace(xlim[0],xlim[1],nsteps+1)
    h = x_arr[1] - x_arr[0]
    y_arr = np.zeros(x_arr.shape)
    y_arr[0] = y0
    for i in range(0,len(x_arr)-1):
        y_arr[i+1] = stepper(func, x_arr[i], y_arr[i], h)
    
    return y_arr


if __name__ == '__main__':
    
    #defning the given function
    dydx = lambda x, y: y/(1+x**2)
    c0 = 1/(np.exp(np.arctan(-20))) #using initial value given
    yfunc = lambda x: c0*np.exp(np.arctan(x))


    xvals1 = np.linspace(-20,20,201) #200 steps
    y_true1 = yfunc(xvals1)

    #first with step of h
    y_est1 = solve(dydx, [-20,20], 200, 1, rk4_step)
    stepsize = 40/200
    rmserr1 = np.sqrt(np.mean((y_true1-y_est1)**2))
    maxerr1 = np.max(np.abs(y_true1-y_est1))
    print(f"Number of steps: {200}\nNumber of function evaluations: {200*4}")
    print(f"step size is {stepsize:4.2f}, so Max predicted error (at the end of integration) from RK4 is of the order {stepsize**4/120:4.2e}")
    print(f"The RMS error in estimation using RK4 of stepsize h is :{rmserr1:4.2e}\nand the max error is: {maxerr1:4.2e}\n")

    #repeat the same thing with RK4-double but with only 73 steps, so that number of function evaluations are the same

    # 4 * 200 ~ 11 * 73
    nsteps = 73
    xvals2 = np.linspace(-20,20,nsteps+1) #200 steps
    y_true2 = yfunc(xvals2)

    y_est2 = solve(dydx, [-20,20], nsteps, 1, rk4_stepd)
    stepsize = 40/nsteps

    rmserr2 = np.sqrt(np.mean((y_true2-y_est2)**2))
    maxerr2 = np.max(np.abs(y_true2-y_est2))
    print(f"Number of steps: {nsteps}\nNumber of function evaluations: {nsteps*11}")
    print(f"step size is {stepsize:4.2f}, so Max predicted error (at the end of integration) from RK4 double-stepper is of the order {2*(stepsize/2)**5/720:4.2e} ")
    print(f"The RMS error in estimation using RK4 of stepsize h is :{rmserr2:4.2e}\nand the max error is: {maxerr2:4.2e}\n")


    fig, ax = plt.subplots(1,2)
    fig.set_size_inches(18,12)
    ax[0].set_title('Solution with simple RK4 -- nsteps=200')
    ax[0].plot(xvals1, np.abs(y_true1-y_est1),'r--', label="err")
    leg = ax[0].legend()
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('err[y(x)]')
    ax[0].grid(True)

    ax[1].set_title('Solution with RK4 double steps -- nsteps=73')
    ax[1].plot(xvals2, np.abs(y_true2-y_est2),'b--', label="err")
    leg = ax[1].legend()
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('err[y(x)]')
    ax[1].grid(True)
    fig.savefig('./err_RK4.png')


        
    