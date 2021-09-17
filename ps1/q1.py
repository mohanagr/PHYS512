import numpy as np # linear algebra


# in this question analytical derivative is used as the function form provided, and it's easy to differentiate


# what to do when function form is not known/ or when not easy to differentiate is handled in Question 2

def get_derivative(func, x0, h):
        

        # 4-th order accurate derivative
        f_prime = (8*(func(x0+h) - func(x0-h)) - (func(x0+2*h) - func(x0-2*h)))/(12*h)
        
        return f_prime


def diff(func, func5, x0):

    '''
    differentiate a function and return the derivate AND the error on the derivative

    input
    func: python function for input
    func5: python function for 5th derivative of input
    x0: float where value required
    '''
    
    eps_m = 2**-52 #machine precision for 64 bit CPU -- 52 bits of mantissa
    optimal_dx = (45*0.25*eps_m*func(x0)/func5(x0))**(1/5)
    print(f"optimal value of dx is {optimal_dx}")
    
    def get_err(fx0, h):
        
        
        err = 1.5*eps_m/h + h**4/30   # this simplies to eps_m**(4/5) * f**(4/5) * f'''''(1/5)
        
        return err
    
    df = get_derivative(func, x0, optimal_dx)
    
    exp_err = get_err(func(x0), optimal_dx)
    
    return [df, exp_err]

if __name__ == '__main__':
    
    # first for exp(x)

    func = np.exp
    func5 = np.exp
    x0=1


    df, err = diff(func,func5,x0)

    true_df = func(1)
    exp_err = np.abs(true_df - df)   # derivative of exp(x) = exp(x)

    print(f"The expected error is {exp_err:5.2e} and the predicted error is {err:5.2e}")

    # --output -- #
    # The expected error is 3.13e-13 and the predicted error is 8.16e-13


    rel_err1 = np.abs(df-true_df)/true_df

    # Now for exp(0.01x)

    func = lambda x: np.exp(0.01*x)
    func5 = lambda x: 10e-10*np.exp(0.01*x)
    x0=1

    df, err = diff(func,func5,x0)

    true_df = func(1)
    exp_err = np.abs(true_df - df)   # derivative of exp(x) = exp(x)

    print(f"The expected error is {exp_err:5.2e} and the predicted error is {err:5.2e}")

    # --output -- #
    # The expected error is 6.87e-16 and the predicted error is 4.80e-15


    rel_err2 = np.abs(df-true_df)/true_df


    # relative error is ~ eps_m**(4/5) in BOTH cases, as f^4 * f'(5) / f' = 1 irrespective of the constant c in exp(c * x)

    # we expect both to be similar and comparable to eps_m**4/5

    # THIS IS INDEED THE CASE

    print(rel_err1, rel_err2, 2**(-52*4/5))

    # 1.1517675969668543e-13 6.801152248406803e-14 3.0002136344885295e-13


