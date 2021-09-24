import numpy as np
import matplotlib.pyplot as plt


def intgadpt(func, a, b, extra=None, tol=1.e-6, which=None, debug=False):
    """Adaptive Step Size integration using Simpson's rule, using memoization of function calls.

    Global variables
    ----------------
    funcalls : int
        number of times parameter function func will be called in this function
    saves : int
        number of times a func call was saved because of memoization

    Parameters
    ----------
    func : function
        python function to integrate. Should return a float or array of floats
    a: float
        lower limit of integration
    b: float
        upper limit of integration
    extra: dict
        A dictionary with key as x and value as func(x)
    tol: float
        absolute error tolerance
    which: str
        For debug usage. Indicates if function output coming from left call or right call.
    debug: bool
        If True, outputs debug information. Quite helpful in understanding the working of this function


    Returns
    -------
    float
        Integral of func [a,b]
    """
    global funcalls, saves
    if(debug):
        print(f"from {which}, limits {a} to {b}")
    if(not extra):
        extra  = {}
        x=np.linspace(a,b,5)
        if(debug):
            print(f"x in use {x}")
        y=np.zeros(len(x)) # we'll calc for all x values since we are just starting up
        for i, xi in enumerate(x):
            
            y[i] = func(xi)
            funcalls+=1
            extra[xi] = y[i]

    else:
        x=np.linspace(a,b,5)
        if(debug):
            print(f"x in use {x}")
        y=np.zeros(len(x))
        # since these are nested calls, we'll first check if we already calculated func value in previous calls
        for i, xi in enumerate(x):
            if xi in extra:
                if(debug):
                    print(f'value {xi} found in previous')
                y[i] = extra[xi]
                saves+=1
            else:
                y[i] = func(xi)
                extra[xi] = y[i]
                funcalls+=1
          
    dx=x[1]-x[0]

    area1=2*dx*(y[0]+4*y[2]+y[4])/3 
    area2=dx*(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/3
    err=np.abs(area1-area2)
    if(debug):
        print(f"error in this call is {err}")
    if err<tol:
        return area2
        
    else:
        xmid=(a+b)/2
        left=intgadpt(func,a,xmid,extra,tol/2, "left",debug)
        right=intgadpt(func,xmid,b,extra,tol/2, "right",debug)
        return left+right


if __name__ == '__main__':
    
    # let us try our memoized function and see:
    # 1. it's accuracy
    # 2. how many function calls it saves us

    # first let's set the function call counter and save counter to 0
    global funcalls, saves
    funcalls = 0
    saves = 0

    func = np.exp
    a = 0
    b = 1
    truth = np.exp(1)-np.exp(0)
    I = intgadpt(func,a,b,debug=False)
    print(f"*** Integating exp() from {a} to {b} ***")
    print(f"Accuracy is {np.abs(I-truth):5.2e}")
    print(f"Number of times function called {funcalls}\nNumber of times we didn't need to call it: {saves}")
    print(f"Overall {saves*100/(saves+funcalls):2.0f}% reduction in function calls compared to lazy way")
    print("")


    # Now let's see it work for our dear Lorentzian
    funcalls = 0
    saves = 0

    func = lambda x: 1/(1+x**2)
    a = -1
    b = 1
    truth = np.arctan(1)-np.arctan(-1)
    I = intgadpt(func,a,b,debug=False)
    print(f"*** Integating Lorentzian from {a} to {b} ***")
    print(f"Accuracy is {np.abs(I-truth):5.2e}")
    print(f"Number of times function called {funcalls}\nNumber of times we didn't need to call it: {saves}")
    print(f"Overall {saves*100/(saves+funcalls):2.0f}% reduction in function calls compared to lazy way")
    print("")




