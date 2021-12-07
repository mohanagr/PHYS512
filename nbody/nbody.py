import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#select all occurence shift ctrl alt j
# repeat line ctrl d

np.random.seed(42)

@nb.njit(parallel=True)
def get_grad(x,pot,RES):
    PE = 0
    nside = pot.shape[0]
    npart = x.shape[0]
    grad = np.zeros((npart,2))
    for i in nb.prange(npart):
        # print("Particle positions:", x[i])
        irow = int(nside // 2 - x[i, 1] // RES - 1)  # row num is given by y position up down
        # but y made to vary 16 to -16 down. [0] is 16
        icol = int(nside // 2 + x[i, 0] // RES)  # col num is given by x position left right
        # print("In indices ffrom grad", irow, icol, "actually", rho[irow,icol])
        # print(np.where(rho>0))
        #         print("pot 1 term", pot[(ix+1)%nside,iy])
        #         print("pot 2 term", pot[ix-1,iy])
        grad[i, 1] = 0.5 * (pot[(irow - 1), icol] - pot[(irow + 1) % nside, icol]) / RES
        # y decreases with high row num.
        grad[i, 0] = 0.5 * (pot[irow, (icol + 1) % nside] - pot[irow, icol - 1]) / RES
        PE += pot[irow, icol]
        # print(self.grad)
    return -grad, PE

@nb.njit
def hist2d(x, mat, RES):
    nside = mat.shape[0]
    for i in range(x.shape[0]):
        irow = int(nside // 2 - x[i, 1] // RES - 1)  # row num is given by y position up down
        # but y made to vary 16 to -16 down. [0] is 16
        icol = int(nside // 2 + x[i, 0] // RES)
        mat[irow,icol]+=1

class nbody():

    def __init__(self, npart, xmax, xmin, nside=1024,soft=1):
        self.nside=nside
        self.x=np.zeros([npart,2])
        self.f=np.zeros([npart,2])
        self.v=np.zeros([npart,2])
        self.grad=np.zeros([npart,2])
        self.m=np.ones(npart)
        self.kernel=None
        self.kernelft=None
        self.npart=npart
        self.rho=np.zeros([self.nside,self.nside])
        self.rhoft = None
        self.pot=np.zeros([self.nside,self.nside])
        self.soft=soft
        self.XMAX = xmax
        self.XMIN = xmin
        self.RES = (self.XMAX-self.XMIN)/self.nside
        self.fac = 1

        # set up the self.kernel here if soft length remains constant
        self.set_kernel()

    def set_kernel(self):
        print("Setting up kernel of Nside", self.nside)
        vec = np.fft.fftfreq(self.nside) * self.nside
        X, Y = np.meshgrid(vec, vec)
        self.kernel = (X ** 2 + Y ** 2)
        # print(self.kernel)
        self.kernel[self.kernel <= self.soft ** 2] = self.soft ** 2
        self.kernel = -(self.RES * self.RES * self.kernel) ** -0.5
        # print(self.kernel)
        self.kernelft = np.fft.rfft2(self.kernel)

    def ic_1part(self):
        # place one particle at (0,0) with zero velocity
        self.x[0,0] = -4
        self.x[0,1] = -4
        self.v[0,0] = 0
        self.v[0,1] = 0

    def ic_2part_circular(self):
        self.x[0, 0] = 0
        self.x[0, 1] = 2

        self.x[1, 0] = 0
        self.x[1, 1] = -2

        self.v[0, 0] = np.sqrt(1/8)
        self.v[1, 0] = -np.sqrt(1/8)

        self.v[0, 1] = 0.2
        self.v[1, 1] = 0.2


    def ic_gauss(self):
        self.x[:] = np.random.randn(self.npart, 2) * self.XMAX / 4  # so that 95% within XMIN - XMAX

    def ics_2gauss(self):
        self.x[:]=np.random.randn(self.npart,2)* self.XMAX / 4
        self.x[:self.npart//2,1]=self.x[:self.npart//2,1]-50
        self.x[self.npart//2:,1]=self.x[self.npart//2:,1]+50
        self.v[:self.npart//2,0]=3 # 3 for 5k
        self.v[self.npart//2:,0]=-3

    def ic_3part(self):
        self.x[0,:] = [0,2]
        self.x[1,:] = [-2, -2]
        self.x[2,:] = [2, -2]



    def enforce_period(self, x):
        x[np.logical_or(x > self.XMAX, x < self.XMIN)] = x[np.logical_or(x > self.XMAX, x < self.XMIN)] % (self.XMAX - self.XMIN)
        x[x > self.XMAX] = x[x > self.XMAX] - (self.XMAX - self.XMIN)

    def set_grad(self):
        PE = 0
        for i in range(self.npart):
            # print("Particle positions:", self.x[i])
            irow = int(self.nside // 2 - self.x[i, 1] // self.RES -1) # row num is given by y position up down
            # but y made to vary 16 to -16 down. [0] is 16
            icol = int(self.nside // 2 + self.x[i, 0] // self.RES) # col num is given by x position left right
            # print("In indices ffrom grad", irow, icol, "actually", self.rho[irow,icol])
            # print(np.where(self.rho>0))
            #         print("pot 1 term", pot[(ix+1)%self.nside,iy])
            #         print("pot 2 term", pot[ix-1,iy])
            self.grad[i, 1] = 0.5 * (self.pot[(irow - 1), icol] - self.pot[(irow+1)%self.nside, icol]) / self.RES
            # y decreases with high row num.
            self.grad[i, 0] = 0.5 * (self.pot[irow, (icol + 1) % self.nside] - self.pot[irow, icol - 1]) / self.RES
            PE += self.pot[irow,icol]
            # print(self.grad)
        return PE

    def update_rho(self, x):
        bins = self.RES*(np.arange(self.nside+1)-self.nside//2) #can equivalentlly use fftshift
        # since each cell is a bin, we have Nside bins, need Nside+1 points

        self.enforce_period(x)
        # self.rho, xedges, yedges = np.histogram2d(x[:, 1], x[:, 0], bins=bins)
        # self.rho = np.flipud(self.rho).copy()
        self.rho[:] = 0
        hist2d(x,self.rho,self.RES)
        # print(bins, self.RES)
        # we want y as top down, x as left right, and y starting 16 at top -16 at bottom
        self.rhoft = np.fft.rfft2(self.rho)

    def update_pot(self):
        self.pot = np.fft.irfft2(self.rhoft * self.kernelft, [self.nside, self.nside])

    def update_forces(self):
        self.f[:], PE = get_grad(self.x, self.pot, self.RES)
        return PE

    def run_leapfrog(self, dt=1, verbose=False):

        self.x[:] = self.x[:] + dt * self.v
        self.update_rho(self.x)
        self.update_pot()
        crap = self.update_forces()
        PE = np.sum(obj.rho * obj.pot)
        self.v[:]=self.v[:]+self.f*dt
        KE = 0.5*np.sum(self.v**2)
        if(verbose):
            print("PE", 0.5*PE, "crap", crap, "KE", KE, "Total E", 0.5*PE+KE, "vs", PE+KE)
        return 0.5*PE+KE


    def run_rk4(self,dt=1,verbose=False):

        x0 = self.x.copy()
        v0 = self.v.copy()
        self.update_rho(x0)
        self.update_pot()
        k1v, PE = get_grad(x0,self.pot,self.RES) # this gives current PE # v terms in accn unit
        KE = 0.5 * np.sum(v0**2)

        if (verbose):
            print("PE", PE, "KE", KE, "Total E", PE + KE)
        k1x = v0 # x terms in vel unit
        xx1=x0+k1x*dt/2
        self.update_rho(xx1)
        self.update_pot()
        k2v, crap = get_grad(xx1, self.pot, self.RES)
        k2x = v0 + k1v*dt/2

        xx2=x0+k2x*dt/2
        self.update_rho(xx2)
        self.update_pot()
        k3v, crap = get_grad(xx2, self.pot, self.RES)
        k3x = v0 + k2v*dt/2

        xx3=x0+k3x*dt
        self.update_rho(xx3)
        self.update_pot()
        k4v, crap = get_grad(xx3, self.pot, self.RES)
        k4x = v0 + k3v*dt

        self.v[:] = self.v + (k1v+2*k2v+2*k3v+k4v)*dt/6
        self.x[:] = self.x + (k1x+2*k2x+2*k3x+k4x)*dt/6
        return 0.5*PE + KE






if(__name__=="__main__"):
    obj = nbody(5000,128,-128,soft=1,nside=128)

    # obj.ic_1part()
    # obj.ic_2part_circular()
    # obj.run(dt=0)
    # print(obj.grad)
    obj.ics_2gauss() #to use this change lims to +/- 128 and particles to 5000
    # obj.ic_3part()
    # obj.ic_gauss()
    # plt.ion()
    # plt.imshow(obj.pot)
    # plt.show()
    # plt.pause(100)

    frames = []  # for storing the generated images
    # fig = plt.figure()
    fac=1
    TE_old=0
    for i in range(20000):
        if(i%100==0):
            verbosity=True
        else:
            verbosity=False

        TE = obj.run_leapfrog(dt=0.005/fac,verbose=verbosity)

        if(TE_old!=0):
            # print(np.abs((TE-TE_old)/TE_old))
            if(np.abs(TE-TE_old)/TE_old > 0.1):
                print('changing')
                fac=fac*2


        # print(-obj.grad)
        # plt.imshow(obj.pot)
        # plt.pause(50)
        # break
        # print(obj.grad)
        # plt.imshow(obj.pot)
        # plt.show()
        # plt.pause(5)
        #
        # break
        if(i%100==0):
            # print("iter", i)
            # if (TE_old != 0):
            #     # print(np.abs((TE-TE_old)/TE_old))
            #     if (np.abs((TE - TE_old) / TE_old) > 0.005):
            #         # plt.pause(10)
            #         print('changing')
            #         fac = fac * 2
                # print(np.abs((TE - TE_old) / TE_old))
            plt.clf()
            plt.imshow(obj.rho)
            plt.pause(0.05)
            # frames.append([plt.imshow(obj.rho,animated=True)])
            # if(len(frames)>50):
            #     break
            TE_old = TE


    # ani = animation.ArtistAnimation(fig, frames, interval=200, blit=True,
    #                                 repeat_delay=500)
    # ani.save('./5000_part_2gauss.gif', dpi=80, writer='imagemagick')
