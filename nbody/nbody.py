import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
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
        # print(self.grad)
    return -grad

@nb.njit
def hist2d(x, mat, RES):
    # this is better than numpy.hist because of parallelization
    nside = mat.shape[0]
    # temp = np.zeros((nside, nside))
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
        self.cur_f = np.zeros([npart, 2]) # these are for leapfrog v2
        self.new_f = np.zeros([npart, 2]) # ------ DITTO -----------
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
        self.x[0,0] = 0
        self.x[0,1] = 0
        self.v[0,0] = 0.2
        self.v[0,1] = 0.05

    def ic_2part_circular(self):
        self.x[0, 0] = 0
        self.x[0, 1] = 2

        self.x[1, 0] = 0
        self.x[1, 1] = -2

        self.v[0, 0] = np.sqrt(1/8)
        self.v[1, 0] = -np.sqrt(1/8)

        # self.v[0, 1] = 0.2
        # self.v[1, 1] = 0.2


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

    def set_grad(self):
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
        self.f[:] = get_grad(self.x, self.pot, self.RES)
        # return get_grad(self.x, self.pot, self.RES)

    def run_leapfrog(self, dt=1, verbose=False):
        KE = 0.5 * np.sum(self.v ** 2)
        PE1 = 0.5 * np.sum(obj.rho * obj.pot)
        self.x[:] = self.x[:] + dt * self.v
        self.update_rho(self.x)
        self.update_pot()
        PE2 = 0.5*np.sum(obj.rho * obj.pot)
        self.update_forces()
        self.v[:]=self.v[:]+self.f*dt

        PE = 0.5*(PE1+PE2)
        if(verbose):
            print("PE", 0.5*PE, "crap", crap, "KE", KE, "Total E", 0.5*PE+KE)
        return PE+KE

    def run_leapfrog2(self, dt=1, verbose=False):

        PE = 0.5 * np.sum(self.rho * self.pot)
        KE = 0.5*np.sum(self.v[:]**2)
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


        self.x[:] = self.x[:] + dt * self.v[:] + self.cur_f * dt * dt /2
        self.update_rho(self.x)
        self.update_pot()
        self.new_f[:] = get_grad(self.x, self.pot, self.RES)
        self.v[:] = self.v[:] + (self.cur_f + self.new_f)*dt/2
        self.cur_f[:] = self.new_f[:]
        return PE+KE


    def run_rk4(self,dt=1,verbose=False):

        x0 = self.x.copy()
        v0 = self.v.copy()
        self.update_rho(x0)
        self.update_pot()
        k1v = get_grad(x0,self.pot,self.RES) # this gives current PE # v terms in accn unit
        KE = 0.5 * np.sum(v0**2)
        PE = 0.5 * np.sum(self.rho * self.pot)

        if (verbose):
            print("PE", PE, "KE", KE, "Total E", PE + KE)
        k1x = v0 # x terms in vel unit
        xx1=x0+k1x*dt/2
        self.update_rho(xx1)
        self.update_pot()
        k2v = get_grad(xx1, self.pot, self.RES)
        k2x = v0 + k1v*dt/2

        xx2=x0+k2x*dt/2
        self.update_rho(xx2)
        self.update_pot()
        k3v = get_grad(xx2, self.pot, self.RES)
        k3x = v0 + k2v*dt/2

        xx3=x0+k3x*dt
        self.update_rho(xx3)
        self.update_pot()
        k4v = get_grad(xx3, self.pot, self.RES)
        k4x = v0 + k3v*dt

        self.v[:] = self.v + (k1v+2*k2v+2*k3v+k4v)*dt/6
        self.x[:] = self.x + (k1x+2*k2x+2*k3x+k4x)*dt/6
        return PE + KE






if(__name__=="__main__"):
    NSIDE = 256
    XMAX=128
    XMIN=-128
    RES=(XMAX-XMIN)/NSIDE
    NPART = 1
    SOFT = 1
    TSTEP = SOFT * RES / np.sqrt(NPART)  # this is generally smaller than eps**3/2
    obj = nbody(NPART,XMAX,XMIN,soft=SOFT,nside=NSIDE)


    obj.ic_1part()
    # obj.ic_2part_circular()
    # obj.run(dt=0)
    # print(obj.grad)
    # v = np.sqrt(NPART*0.6/XMAX)
    # obj.ics_2gauss(v) #to use this change lims to +/- 128 and particles to 5000
    # obj.ic_3part()
    # obj.ic_gauss()
    obj.update_rho(obj.x)
    obj.update_pot()
    obj.cur_f[:] = get_grad(obj.x,obj.pot,obj.RES)
    frames = []  # for storing the generated images
    fig = plt.figure()
    fac=1
    logfile = open('./dump.txt', 'w')

    try:
        for i in range(20000):

            TE = obj.run_leapfrog2(dt=TSTEP)
            print(obj.x)
            print(TE)
            if(i%10==0):
                # print("iter", i)
                # if (TE_old != 0):
                #     # print(np.abs((TE-TE_old)/TE_old))
                #     if (np.abs((TE - TE_old) / TE_old) > 0.005):
                #         # plt.pause(10)
                #         print('changing')
                #         fac = fac * 2
                    # print(np.abs((TE - TE_old) / TE_old))
                #update TE_old=TE at the end
                logfile.write(str(TE)+"\n")
                # plt.clf()
                frames.append([plt.imshow(obj.rho**0.5,cmap='inferno',animated=True)])
                # plt.colorbar()
                no_labels = 9  # how many labels to see on axis x
                step = int(NSIDE / (no_labels - 1))  # step between consecutive labels
                positions = np.arange(0, NSIDE + 1, step)  # pixel count at label position
                labels = -positions*RES+XMAX  # labels you want to see --- imshow plots origin at top
                plt.yticks(positions, labels)
                plt.xticks(positions, labels)
                plt.pause(0.5)
    except KeyboardInterrupt as e:
        pass
    finally:
        ani = animation.ArtistAnimation(fig, frames, interval=190, blit=True,
                                        repeat_delay=500)
        ani.save('./dump.gif', dpi=80, writer='imagemagick')
        logfile.close()



