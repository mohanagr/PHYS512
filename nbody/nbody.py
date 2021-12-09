import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
#select all occurence shift ctrl alt j
# repeat line ctrl d

np.random.seed(42)

# @nb.njit(parallel=True)
def get_grad(x,pot,RES,periodic=True):
    PE = 0
    nside = pot.shape[0]
    npart = x.shape[0]
    grad = np.zeros((npart,2))

    for i in nb.prange(npart):
        # print("Particle positions:", x[i])
        # if(np.abs(x[i,0])>RES*nside//2): x[i,0] = np.round(x[i,0]) # save yourself some roundoff errors
        # if (np.abs(x[i, 1]) > RES * nside // 2): x[i, 1] = np.round(x[i, 1])
        #
        irow = int(nside /2 - x[i, 1]/RES)  # row num is given by y position up down
        # but y made to vary 16 to -16 down. [0] is 16
        icol = int(nside /2 + x[i, 0]/RES) # col num is given by x position left right

        # irow = int(nside // 2 - x[i, 1] // RES - 1)  # row num is given by y position up down
        # # but y made to vary 16 to -16 down. [0] is 16
        # icol = int(nside // 2 + x[i, 0] // RES)
        # print("In indices ffrom grad", irow, icol, "actually", rho[irow,icol])
        # print(np.where(rho>0))
        #         print("pot 1 term", pot[(ix+1)%nside,iy])
        #         print("pot 2 term", pot[ix-1,iy])
        # print(x[i, 0], x[i, 1])
        # print(irow, icol)

        if(periodic):
            grad[i, 1] = 0.5 * (pot[(irow - 1), icol] - pot[(irow + 1) % nside, icol]) / RES
            # y decreases with high row num.
            grad[i, 0] = 0.5 * (pot[irow, (icol + 1) % nside] - pot[irow, icol - 1]) / RES

        else:
            if(icol==0 and irow==0):
                grad[i, 1] = (pot[irow, icol] - pot[irow+1, icol]) / RES # Y DECREASES WITH INDEX phi(y) - phi(y-h)
                grad[i, 0] = (pot[irow, icol + 1] - pot[irow, icol]) / RES  #phi(x+h) - phi(x)
            elif(icol==(nside-1) and irow==0):
                grad[i, 1] = (pot[irow, icol] - pot[irow + 1, icol]) / RES
                grad[i, 0] = (pot[irow, icol] - pot[irow, icol - 1]) / RES #phi(x) - phi(x-h)
            elif(icol==0 and irow==(nside-1)):
                grad[i, 1] = (pot[irow - 1, icol] - pot[irow, icol]) / RES #phi(y+h) - phi(y)
                grad[i, 0] = (pot[irow, icol + 1] - pot[irow, icol]) / RES
            elif(icol==(nside-1) and irow==(nside-1)):
                grad[i, 1] = (pot[irow - 1, icol] - pot[irow, icol]) / RES #phi(y+h) - phi(y)
                grad[i, 0] = (pot[irow, icol] - pot[irow, icol - 1]) / RES #phi(x) - phi(x-h)
            elif(icol==(nside-1) and irow>0):
                grad[i, 1] = 0.5 * (pot[(irow - 1), icol] - pot[(irow + 1) % nside, icol]) / RES
                grad[i, 0] = (pot[irow, icol] - pot[irow, icol - 1]) / RES #phi(x) - phi(x-h)
            elif (icol > 0 and irow==(nside-1)):
                grad[i, 1] = (pot[irow - 1, icol] - pot[irow, icol]) / RES
                grad[i, 0] = 0.5 * (pot[irow, (icol + 1) % nside] - pot[irow, icol - 1]) / RES
            elif (icol==0 and irow>0):
                grad[i, 1] = 0.5 * (pot[(irow - 1), icol] - pot[(irow + 1) % nside, icol]) / RES
                grad[i, 0] = (pot[irow, icol + 1] - pot[irow, icol]) / RES
            elif (icol>0 and irow==0):
                grad[i, 1] = (pot[irow, icol] - pot[irow + 1, icol]) / RES
                grad[i, 0] = 0.5 * (pot[irow, (icol + 1) % nside] - pot[irow, icol - 1]) / RES
            else:
                grad[i, 1] = 0.5 * (pot[(irow - 1), icol] - pot[(irow + 1) % nside, icol]) / RES
                # y decreases with high row num.
                grad[i, 0] = 0.5 * (pot[irow, (icol + 1) % nside] - pot[irow, icol - 1]) / RES
    # print("from grad", grad)
    return -grad

@nb.njit
def hist2d(x, mat, RES):
    # this is better than numpy.hist because of parallelization
    nside = mat.shape[0]
    # print(nside)
    # print(mat)
    # sys.exit(0)
    # temp = np.zeros((nside, nside))
    # print(x)
    for i in range(x.shape[0]):
        # print("inside for loop")
        irow = int(nside / 2 - x[i, 1] / RES)  # row num is given by y position up down
        # but y made to vary 16 to -16 down. [0] is 16
        icol = int(nside / 2 + x[i, 0] / RES)
        # print(irow,icol)
        mat[irow,icol]+=1
    # print(mat)


class nbody():

    def __init__(self, npart, xmax, xmin, nside=1024,soft=1,periodic=True):
        self.nside=nside
        self.x=np.zeros([npart,2])
        self.f=np.zeros([npart,2])
        self.cur_f = np.zeros([npart, 2]) # these are for leapfrog v2
        self.new_f = np.zeros([npart, 2]) # ------ DITTO -----------
        self.v=np.zeros([npart,2])
        self.grad=np.zeros([npart,2])
        self.m=np.ones(npart)
        self.kernel=None
        self.kernelft=None
        self.npart=npart
        self.rhoft = None
        self.soft=soft
        self.XMAX = xmax
        self.XMIN = xmin
        self.RES = (self.XMAX-self.XMIN)/self.nside
        self.fac = 1
        self.mask = np.ones(npart,dtype='bool')
        self.periodic = periodic
        if (periodic):
            self.rho = np.zeros([self.nside, self.nside])
            self.pot = np.zeros([self.nside, self.nside])
        else:
            self.rho = np.zeros([2 * self.nside, 2 * self.nside])
            self.pot = np.zeros([2 * self.nside, 2 * self.nside])
        # set up the self.kernel here if soft length remains constant
        self.set_kernel()

    def set_kernel(self):
        print("Setting up kernel of Nside", self.nside)
        if (self.periodic):
            vec = np.fft.fftfreq(self.nside) * self.nside
        else:
            vec = np.fft.fftfreq(2*self.nside) * 2*self.nside

        X, Y = np.meshgrid(vec, vec)
        self.kernel = (X ** 2 + Y ** 2)
        # print(k)
        self.kernel[self.kernel <= self.soft ** 2] = self.soft ** 2
        self.kernel = -(self.RES * self.RES * self.kernel) ** -0.5
        # print(k)
        #
        # if(self.periodic):
        #     self.kernel = k.copy()
        # else:
        #     self.kernel = np.zeros((2*self.nside,2*self.nside))
        #     self.kernel[:self.nside,:self.nside] = k.copy()
        self.kernelft = np.fft.rfft2(self.kernel)



    def ic_1part(self):
        # place one particle at (0,0) with zero velocity
        self.x[0,0] = 0
        self.x[0,1] = 0
        self.v[0,0] = 0
        self.v[0,1] = 0.2

    def ic_2part_circular(self):
        self.x[0, 0] = 0
        self.x[0, 1] = 1

        self.x[1, 0] = 0
        self.x[1, 1] = -1

        self.v[0, 0] = np.sqrt(1/4)
        self.v[1, 0] = -np.sqrt(1/4)

        self.v[0, 1] = 0.2
        self.v[1, 1] = 0.2


    def ic_gauss(self):
        self.x[:] = np.random.randn(self.npart, 2) * self.XMAX / 4  # so that 95% within XMIN - XMAX

    def ics_2gauss(self, vel):
        self.x[:]=np.random.randn(self.npart,2)*self.XMAX/4
        self.x[:self.npart//2,1]=self.x[:self.npart//2,1]-self.XMAX/2.4
        self.x[self.npart//2:,1]=self.x[self.npart//2:,1]+self.XMAX/2.4
        self.v[:self.npart//2,0]= vel # 3 for 5k
        self.v[self.npart//2:,0]=-vel

    def ic_3part(self):
        self.x[0,:] = [0,2]
        self.x[1,:] = [-2, -2]
        self.x[2,:] = [2, -2]



    def enforce_period(self, x):
            x[np.logical_or(x > self.XMAX, x < self.XMIN)] = x[np.logical_or(x > self.XMAX, x < self.XMIN)] % (self.XMAX - self.XMIN)
            x[x > self.XMAX] = x[x > self.XMAX] - (self.XMAX - self.XMIN)
            # print("from enforce", x.shape, x[:,0].min(), x[:,0].max())
            return x

    def update_rho(self):
        bins = self.RES*(np.arange(self.nside+1)-self.nside//2) #can equivalentlly use fftshift
        # since each cell is a bin, we have Nside bins, need Nside+1 points

        if(self.periodic):
            self.x = self.enforce_period(self.x) # had to figure out the hard way : with fancy indexing numpy doesnt always return a view, it's a copy
        else:
            # print("NP WHERE",np.where(np.logical_or(self.x > self.XMAX, self.x < self.XMIN))[0])
            self.mask[np.where(np.logical_or(self.x > self.XMAX, self.x < self.XMIN))[0]] = False

        # print("after enforce", self.x.shape, self.x[:, 0].min(), self.x[:, 0].max())
        # self.rho, xedges, yedges = np.histogram2d(x[:, 1], x[:, 0], bins=bins)
        # self.rho = np.flipud(self.rho).copy()
        self.rho[:] = 0

        hist2d(self.x[self.mask],self.rho[:self.nside,:self.nside],self.RES)
        # print(bins, self.RES)
        # we want y as top down, x as left right, and y starting 16 at top -16 at bottom

        self.rhoft = np.fft.rfft2(self.rho)
        # print(self.rho.shape,self.rhoft.shape,self.kernelft.shape)

    def custom_update_rho(self,x,update_mask=False):
        '''
        This method is for Runge-Kutta.
        We don't want to update the mask at intermediate steps
        '''
        bins = self.RES*(np.arange(self.nside+1)-self.nside//2) #can equivalentlly use fftshift
        # since each cell is a bin, we have Nside bins, need Nside+1 points

        if(self.periodic):
            x = self.enforce_period(x) # had to figure out the hard way : with fancy indexing numpy doesnt always return a view, it's a copy
        else:
            if(update_mask):
                # print("NP WHERE",np.where(np.logical_or(self.x > self.XMAX, self.x < self.XMIN))[0])
                self.mask[np.where(np.logical_or(x > self.XMAX, x < self.XMIN))[0]] = False

        # print("after enforce", self.x.shape, self.x[:, 0].min(), self.x[:, 0].max())
        # self.rho, xedges, yedges = np.histogram2d(x[:, 1], x[:, 0], bins=bins)
        # self.rho = np.flipud(self.rho).copy()
        self.rho[:] = 0

        hist2d(x[self.mask],self.rho[:self.nside,:self.nside],self.RES)
        # print(bins, self.RES)
        # we want y as top down, x as left right, and y starting 16 at top -16 at bottom

        self.rhoft = np.fft.rfft2(self.rho)
        # print(self.rho.shape,self.rhoft.shape,self.kernelft.shape)

    def update_pot(self):
        # print(self.kernel)
        self.pot = np.fft.irfft2(self.rhoft * self.kernelft)
        # print(self.pot.shape)
        # plt.imshow(self.pot[:self.nside, :self.nside])
        # plt.pause(50)
        # sys.exit(0)

    def update_forces(self):
        self.f[self.mask] = get_grad(self.x[self.mask], self.pot[:self.nside,:self.nside], self.RES,periodic=self.periodic)
        # return get_grad(self.x, self.pot, self.RES)

    def run_leapfrog(self, dt=1, verbose=False):
        KE = 0.5 * np.sum(self.v[self.mask] ** 2)
        PE1 = 0.5 * np.sum(obj.rho[:self.nside,:self.nside] * obj.pot[:self.nside,:self.nside])
        # print("update to x gonna be: ",dt * self.v[self.mask])
        self.x = self.x + dt * self.v
        self.update_rho()
        self.update_pot()
        PE2 = 0.5 * np.sum(obj.rho[:self.nside,:self.nside] * obj.pot[:self.nside,:self.nside])
        self.update_forces()
        # print("FORCE IS",self.f)
        self.v =self.v +self.f*dt

        PE = 0.5*(PE1+PE2)
        if(verbose):
            print("PE", 0.5*PE, "crap", crap, "KE", KE, "Total E", 0.5*PE+KE)
        return PE+KE

    def run_leapfrog2(self, dt=1, verbose=False):

        PE = 0.5 * np.sum(self.rho * self.pot)
        KE = 0.5*np.sum(self.v[self.mask]**2)
        # vhalf = np.zeros((self.npart,2))
        #
        # cur_f = get_grad(self.x, self.pot, self.RES).copy()
        # vhalf = self.v[:] + cur_f * dt/2 # half Euler step to get started
        #
        # self.x[:] = self.x[:] + dt * vhalf[:]
        #
        # self.update_rho(self.x)
        # self.update_pot()
        # new_f = get_grad(self.x, self.pot, self.RES).copy()
        # self.v[:]=self.v[:]+0.5*dt*new_f
        #
        # cur_f[:] = new_f[:]


        self.x[self.mask] = self.x[self.mask] + dt * self.v[self.mask] + self.cur_f * dt * dt /2
        self.update_rho()
        self.update_pot()
        self.new_f = get_grad(self.x[self.mask], self.pot, self.RES)
        self.v[self.mask] = self.v[self.mask] + (self.cur_f + self.new_f)*dt/2
        self.cur_f = self.new_f
        return PE+KE

    def get_derivs(self,xx):
        nn = self.npart
        x = xx[:nn, :]
        v = xx[nn:, :]
        f = np.zeros((self.npart,2))

        self.custom_update_rho(x, update_mask=True) # rho always uses the masked x
        self.update_pot()
        f[self.mask] = get_grad(x[self.mask], self.pot[:self.nside,:self.nside], self.RES)
        return np.vstack([v, f])

    def run_rk4(self,dt=1):
        PE = 0.5 * np.sum(self.rho[:self.nside,:self.nside] * self.pot[:self.nside,:self.nside])
        KE = 0.5 * np.sum(self.v[self.mask] ** 2)
        xx = np.vstack([self.x, self.v])
        k1 = self.get_derivs(xx)
        k2 = self.get_derivs(xx + k1 * dt / 2)
        k3 = self.get_derivs(xx + k2 * dt / 2)
        k4 = self.get_derivs(xx + k3 * dt)

        tot = (k1 + 2 * k2 + 2 * k3 + k4) / 6

        nn = self.npart

        self.x[:] = self.x[:] + tot[:nn, :] * dt
        self.v[:] = self.v[:] + tot[nn:, :] * dt
        return PE+KE


if(__name__=="__main__"):
    NSIDE = 128
    XMAX=16
    XMIN=-16
    RES=(XMAX-XMIN)/NSIDE
    NPART = 50000
    SOFT = 2
    TSTEP = 0.2*SOFT * RES / np.sqrt(NPART)  # this is generally smaller than eps**3/2
    obj = nbody(NPART,XMAX,XMIN,soft=SOFT,nside=NSIDE,periodic=True)


    # obj.ic_1part()
    # obj.ic_2part_circular()
    # obj.run(dt=0)
    # print(obj.grad)
    v = 1.5*np.sqrt(NPART*0.6/XMAX)
    obj.ics_2gauss(v) #to use this change lims to +/- 128 and particles to 5000
    # obj.ic_3part()
    # obj.ic_gauss()
    obj.update_rho()
    obj.update_pot()
    # plt.imshow(obj.rho)
    # plt.pause(10)
    # obj.cur_f[:] = get_grad(obj.x,obj.pot,obj.RES)
    frames = []  # for storing the generated images
    fig = plt.figure()
    fac=1
    logfile = open('./dump.txt', 'w')

    try:
        for i in range(1200):

            TE = obj.run_rk4(dt=TSTEP)
            # print(obj.x)

            if(i%10==0):
                print(TE)
                # print("iter", i)
                # if (TE_old != 0):
                #     # print(np.abs((TE-TE_old)/TE_old))
                #     if (np.abs((TE - TE_old) / TE_old) > 0.005):
                #         # plt.pause(10)
                #         print('changing')
                #         fac = fac * 2
                    # print(np.abs((TE - TE_old) / TE_old))
                #update TE_old=TE at the end
                # print(obj.x)
                logfile.write(str(TE)+"\n")
                # plt.clf()
                frames.append([plt.imshow(obj.rho[:NSIDE,:NSIDE]**0.5,cmap='inferno',animated=True)])
                # plt.colorbar()
                no_labels = 9  # how many labels to see on axis x
                step = int(NSIDE / (no_labels - 1))  # step between consecutive labels
                positions = np.arange(0, NSIDE + 1, step)  # pixel count at label position
                labels = -positions*RES+XMAX  # labels you want to see --- imshow plots origin at top
                plt.yticks(positions, labels)
                plt.xticks(positions, labels)
                plt.pause(0.01)
    except KeyboardInterrupt as e:
        pass
    finally:
        ani = animation.ArtistAnimation(fig, frames, interval=200, blit=True,
                                        repeat_delay=500)
        ani.save('./dump.gif', dpi=80, writer='imagemagick')
        logfile.close()



