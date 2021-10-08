import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class leastsq():
    
    def __init__(self, A,y, N=None):
        
        assert(A.shape[0]==y.shape[0])
        self.A = A
        self.y = y
        self.np = A.shape[1]
        self.nd = A.shape[0]
        self.N = N
    
    def fit(self):
        
        if(self.N is not None):
            #do a cholesky decomposition to recast the eqn in the form A^T*A*m = A^T*d
            L = np.linalg.cholesky(self.N)
            Linv = np.linalg.inv(L)
            self.Anew = Linv@A
            self.ynew = Linv@y
        else:
            self.Anew = self.A
            self.ynew = self.y
        
        u,s,v = np.linalg.svd(self.Anew,0)

        cn = s.max()/s.min()
        if(cn>1.e10):
            warnings.warn("The A matrix is ill-conditioned. Too large condition number. Fit may be crap.")
        s = np.eye(len(s))*s

        rhs = u.T
        lhs = s@v
        self.pinv = np.linalg.inv(lhs)@rhs
        self.params = self.pinv@self.ynew
        self.hat = np.linalg.inv(self.Anew.T@self.Anew)

    
    def get_param_cov(self):
        
        return self.hat
    
    def get_pred_err(self):
        
        covd = self.A@self.hat@self.A.T
        
        return covd

if __name__ == '__main__':
    
    # first let's run a vanilla version of least-squares. error assumed N(0,1) -- standard normal.
    # therefore Noise matrix = Identity

    data = np.loadtxt('./dish_zenith.txt')

    print("*** Identity Noise matrix ***\n")
    A = np.zeros((data.shape[0],4))
    A[:,0] = data[:,0]**2 + data[:,1]**2 # x^2 + y^2
    A[:,1] = data[:,0]
    A[:,2] = data[:,1]
    A[:,3] = 1
    y = data[:,2] # z values

    lsq = leastsq(A,y)
    lsq.fit()
    m = lsq.params

    ypred = A@m
    resd = y-ypred

    rmserr = np.sqrt(np.sum(resd**2)/(y.shape[0]-m.shape[0]+1))
    chisq = resd.T@resd # Noise is identity right now
    a = m[0]
    x0 = -0.5*m[1]/a
    y0 = -0.5*m[2]/a
    z0 = m[3]-x0**2-y0**2
    print(f"params of the paraboloid are:\na\t= {a:4.2e}\nx0\t= {x0:4.2f}\ny0\t= {y0:4.2f}\nz0\t= {z0:4.2f}")
    print(f"Chisq of fit is: {chisq:4.2f}")
    print(f"RMS studentized err is: {rmserr:4.2f} mm")

    # let's estimate a better noise model

    print("\n*** Repeating with a better Noise model (refer readme) ***\n")

    bin_std, bin_edges, bin_num = stats.binned_statistic(y,resd,statistic='std')

    #some nice plotting
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(10,8)
    ax.set_title('Residuals at each "y"')
    ax.plot(y, resd,'b.', label="error")
    ax.axhline(resd.mean(), c='r', label='Global mean')
    for edge in bin_edges:
        c = ax.axvline(edge, c='g',linestyle='--')
    handles, labels = ax.get_legend_handles_labels()
    handles.append(c)
    labels.append('bin edge')
    leg = ax.legend(handles, labels)
    ax.set_xlabel('y[i]')
    ax.set_ylabel('err at y[i]')
    fig.savefig('./residual_variation.png')


    # let's create a new noise matrix with each diagonal element =  square of avg error of each bin

    Nnew = np.eye(y.shape[0])
    for i in range(0, len(bin_edges)-1):
        idx = np.where(np.logical_and(y>bin_edges[i], y<=bin_edges[i+1]))
        Nnew[idx,idx] = bin_std[i]**2
    # set the first element separately
    idx = np.where(y==bin_edges[0])
    Nnew[idx,idx] = bin_std[0]**2

    lsq = leastsq(A,y,Nnew)
    lsq.fit()
    m = lsq.params
    a = m[0]
    x0 = -0.5*m[1]/a
    y0 = -0.5*m[2]/a
    z0 = m[3]-x0**2-y0**2

    ypred = A@m
    resd = y-ypred
    rmserr = np.sqrt(np.mean(resd**2))
    chisq = resd.T@np.linalg.inv(Nnew)@resd
    print(f"params of the paraboloid are:\na\t= {a:4.2e}\nx0\t= {x0:4.2f}\ny0\t= {y0:4.2f}\nz0\t= {z0:4.2f}")
    print(f"Chisq of fit is: {chisq:4.2f}")
    print(f"RMS studentized err is: {rmserr:4.2f} mm")

    mcov = lsq.get_param_cov()
    merr = np.sqrt(np.diag(mcov))

    err_a = merr[0]

    err_focus = np.abs(0.25*err_a/a**2)

    print(f"Error in focus is {err_focus:4.2f} mm")






