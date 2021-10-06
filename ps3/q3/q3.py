import numpy as np
import matplotlib.pyplot as plt

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
    rmserr = np.sqrt(np.mean(resd**2))
    a = m[0]
    x0 = -0.5*m[1]/a
    y0 = -0.5*m[2]/a
    z0 = m[3]-x0**2-y0**2
    print(f"params of the paraboloid are:\na\t= {a:4.2e}\nx0\t= {x0:4.2f}\ny0\t= {y0:4.2f}\nz0\t= {z0:4.2f}")
    print(f"\nRMS error or in this case chisq of fit is: {rmserr:4.2f}")


