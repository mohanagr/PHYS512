import numpy as np
from matplotlib import pyplot as plt


class CustomLog():
    """CustomLog class implements functionality to fit a custom log function of any base using log2"""
    def __init__(self, base=np.e, n=100, order=10, fit='cheb', tol=1.e-6):
        
        self.n = n
        self.order=order
        self.fit_type=fit
        self.base = base
        self.tolerance = tol

        if(self.fit_type=='cheb'):
            self.fit_func = np.polynomial.chebyshev.chebfit
            self.fit_eval_func = np.polynomial.chebyshev.chebval
        elif(self.fit_type=='leg'):
            self.fit_func = np.polynomial.legendre.legfit
            self.fit_eval_func = np.polynomial.legendre.legval
        else:
            raise ValueError("Unknown function type passed for fit")

        self._fit()

    def _fit(self):

        x = np.linspace(0.5,1,self.n)
        y = np.log2(x)
        xnew = 4*x-3 #this goes from -1 to 1, in order to evaluate chebyshevs across their full range, not just half
        self.fit_coeff = self.fit_func(xnew, y, self.order)
        self.fit_val = self.fit_eval_func(xnew, self.fit_coeff)
        self.fit_err = np.sqrt(np.mean((y-self.fit_val)**2))


    def _getSuggestedOrder(self):

        s = 0
        for i in range(0,len(self.fit_coeff)):
            s_prev = s
            s += np.abs(self.fit_coeff[-i-1])
            if s>self.tolerance:
                suggestion = len(self.fit_coeff) - i
                print("***************************** Suggestion mode ON ***********************************")
                print(f"last {i} coefficients can be safely ignored to stay below error tolerance of {self.tolerance:4.2e}")
                self.truncation_err = s_prev
                print(f"Predicted order truncation error is {self.truncation_err:4.2e}")
                print("************************************************************************************")
                break
        return suggestion

    def __call__(self, x, suggest=False):

        m1, exp1 = np.frexp(x)
        m2, exp2 = np.frexp(self.base)
        m1 = 4*m1-3   # we need to rescale our mantissa as the chebyshevs take rescaled arguments
        m2 = 4*m2-3
        if(suggest):
            i = self._getSuggestedOrder()
            # print(self.fit_coeff)
            # print(self.fit_coeff[:i])
            logm1 = self.fit_eval_func(m1, self.fit_coeff[:i])
            logm2 = self.fit_eval_func(m2, self.fit_coeff[:i])
        else:
            # use the full high order function for evaluation
            logm1 = self.fit_eval_func(m1, self.fit_coeff)
            logm2 = self.fit_eval_func(m2, self.fit_coeff)

        return (logm1+exp1)/(logm2+exp2)


if __name__ == '__main__':
    
    # first let's create a callable mylog object from our CustomLog class
    # by default we use 10th order Chebyshev over 100 points to estimate log2(x)
    # and by default we create a function for emulating natural log(x)
    x_test = np.linspace(0.1,10,10)
    y_test = np.log(x_test)


    mylog = CustomLog(order=10)
    y_pred = mylog(x_test)
    err = np.sqrt(np.mean((y_test-y_pred)**2))
    print(f"Error obtained for custom log function by using all orders of Chebyshevs is {err:4.2e}")
    
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(6,4)
    ax.set_title('actual log vs my log -- chebyshev fit')
    ax.plot(np.linspace(0.1,10,1000),np.log(np.linspace(0.1,10,1000)), 'r-', label='numpy log')
    ax.plot(x_test, y_pred,'g*', label='values from mylog')
    leg = ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('log(x)')
    ax.grid(True)
    fig.savefig('./ps2_q3_actuallog_vs_mylog.png')


    # Even though we've fit some high order, we don't really need to use all of them
    # we can stay below our specified error tolerance by using first few orders
    # let's turn on the suggestedOrder feature of our function

    y_pred = mylog(x_test, suggest=True)
    err = np.sqrt(np.mean((y_test-y_pred)**2))
    print(f"Error obtained for custom log function by using only the suggested order of Chebyshevs is {err:4.2e}")


    # now let us repeat this exercise with legendre polynomials instead of Chebyshevs
    mylog = CustomLog(fit="leg", order=10)
    y_pred = mylog(x_test)
    err = np.sqrt(np.mean((y_test-y_pred)**2))
    print(f"\n\nError obtained for custom log function by using all orders of Legendre is {err:4.2e}")

    # Even though we've fit some high order, we don't really need to use all of them
    # we can stay below our specified error tolerance by using first few orders
    # let's turn on the suggestedOrder feature of our function

    y_pred = mylog(x_test, suggest=True)
    err = np.sqrt(np.mean((y_test-y_pred)**2))
    print(f"Error obtained for custom log function by using only the suggested order of Legendre is {err:4.2e}")


    # print("\nVerification of Chebyshevs better than Legendre for high order (50 in this example) fitting")
    # print("---------------------------------------------------------------------------------------------")
    # # uncomment these lines to see the improvement obtained using Chebs over Legs
    # mylog = CustomLog(order=50)
    # y_pred = mylog(x_test)
    # err = np.sqrt(np.mean((y_test-y_pred)**2))
    # print(f"Error obtained for custom log function by using all orders of Chebyshevs is {err:4.2e}")

    # mylog = CustomLog(fit="leg", order=50)
    # y_pred = mylog(x_test)
    # err = np.sqrt(np.mean((y_test-y_pred)**2))
    # print(f"Error obtained for custom log function by using all orders of Legendre is {err:4.2e}\n\n")













