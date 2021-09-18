import numpy as np # linear algebra
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep, CubicSpline


# I suggested this more general method of estimating error in the Tutorial with Rigel, Sep-16


def lakeshore(V, data, frac=0.8):
    '''
    inputs

    V: array of voltage points where interpolated value is desired
    data: 2-D array of datafile (first col T, second col V)
    frac: fraction of original input data to use for resampling, for error estimation. default - 80% data used

    outputs

    2-D array of interpolated Temperature values and error at each value
    '''



    N_evals = 100 # number of times to resample for ensemble average/stddev etc. later
    f = frac
    N_actual = np.shape(data)[0]
    N_rs = int(f * N_actual) # take N_rs points out of total dataset for resampling and error estimation

    rng = np.random.default_rng(seed=30)


    # value estimation

    V_x = data[:,1][::-1]
    T_y = data[:,0][::-1]
    cs = CubicSpline(V_x, T_y)
    result = cs(V)



    # error estimation
    evals = []
    for i in range(N_evals):
        
        idx = np.sort(rng.choice(N_actual,N_rs,replace=False))[::-1] # generate N_rs random indices
        # reverse the sorted order as V is in decreaseing order and CS needs it in increasing

        # what if on of our inputs in x_input is out-of-bound for the subset x_temp?
        # no worries! CubicSpline will return NaN for that particular x value. 
        # and Numpy's mean/stddev etc.ignore these NaN's while computing the statistic.
        # E.g. if x[0] in x_input was out of bound 2 out of 10 evals, our stddev of x[0] will be over 8 values

        
        x_temp = data[idx,1] # voltage is input             # temp = temporary, not temperature
        y_temp = data[idx,0] # temperature is output

        cs = CubicSpline(x_temp, y_temp)
        
        y_interp_output = cs(V)
        evals.append(y_interp_output)
        
    evals = np.array(evals)

    errors = np.std(evals, axis=0)

    # the range of error is between 1.e-5 to 1.e-2, and depends on the regime of the function
    print(f"The mean error in interpolation is: {errors.mean():5.1e}")
    print(f"Min error is {errors.min():5.1e} and Max error is : {errors.max():5.1e}")
    print(f"The spread of interpolation error is: {errors.std():5.1e}")

    return [result, errors]


if __name__ == "__main__":

    data = np.loadtxt("./lakeshore.txt")

    # generate 50 random x values to simulate the input -- points we'd like to know the value of function at
    xmin = data[:,1].min() 
    xmax = data[:,1].max()
    rng = np.random.default_rng(seed=30)
    voltage_input =  np.sort(rng.random(50)*(xmax - xmin) + xmin) 

    T_output, T_err = lakeshore(voltage_input, data)

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(10, 8)

    ax.set_title('given data vs interpolated comparison')
    ax.plot(data[:,1],data[:,0], 'r--', label='given data')
    ax.plot(voltage_input, T_output,'go', label='interpolated values at input')
    leg = ax.legend()
    ax.set_xlabel('Voltage')
    ax.set_ylabel('Temperature')
    ax.grid(True)
    fig.savefig('./fig_output.png')

    print("***** please see comments inside code and README.TXT ******")




