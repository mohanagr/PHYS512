import numpy as np

def ndiff(func, x, full=False):

    eps_m = 2**-52
    
    h = 0.001 # start with default h for estimating third derivative
    y2 = func(x+2*h)
    y1 = func(x+h)
    y0 = func(x)
    ym1 = func(x-h)
    ym2 = func(x-2*h)
    
    df3 = (y2-ym2 - 2*(y1-ym1))/(2*h**3) # third derivative for error estimation and optimal dx
    # this estimate of third derivative is not optimal for an arbitrary h, but we just need a good-enough order-of-magnitude estimate
    
    h = (24*eps_m*np.abs(y0)/np.abs(df3))**(1/3) # get optimal h for first derivative
    
    #recalculate required function evaluations at new h
    y1 = func(x+h)
    ym1 = func(x-h)
    
    dfunc = (y1-ym1)/(2*h)
    
    if full:
        
        err = eps_m**(2/3) * y0**(2/3) * df3**(1/3)
        return dfunc, err
    
    return dfunc


if __name__=="__main__":


    df, err = ndiff(np.exp,0.001, full=True) 
    print(f"The obtained error is: {np.abs(np.exp(0.001)-df):5.2e} and the predicted error is {err:5.2e} ")
    print("***** please see comments inside code ******")
    #  -- output --
    # The obtained error is: 4.99e-11 and the predicted error is 3.67e-11

