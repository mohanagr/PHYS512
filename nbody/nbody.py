import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#select all occurence shift ctrl alt j
# repeat line ctrl d


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


    def ic_gauss(self):
        self.x[:] = np.random.randn(self.npart, 2) * self.XMAX / 4  # so that 95% within XMIN - XMAX

    def ics_2gauss(self):
        self.x[:]=np.random.randn(self.npart,2)* self.XMAX / 4
        self.x[:self.npart//2,1]=self.x[:self.npart//2,1]-50
        self.x[self.npart//2:,1]=self.x[self.npart//2:,1]+50
        self.v[:self.npart//2,0]=3
        self.v[self.npart//2:,0]=-3

    def enforce_period(self):
        self.x[np.logical_or(self.x > self.XMAX, self.x < self.XMIN)] = self.x[np.logical_or(self.x > self.XMAX, self.x < self.XMIN)] % (self.XMAX - self.XMIN)
        self.x[self.x > self.XMAX] = self.x[self.x > self.XMAX] - (self.XMAX - self.XMIN)

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
            # print(self.grad)

    def update_rho(self):
        bins = self.RES*(np.arange(self.nside+1)-self.nside//2) #can equivalentlly use fftshift
        # since each cell is a bin, we have Nside bins, need Nside+1 points

        self.enforce_period()
        self.rho, xedges, yedges = np.histogram2d(self.x[:, 1], self.x[:, 0], bins=bins)
        self.rho = np.flipud(self.rho).copy()
        # print(bins, self.RES)
        # we want y as top down, x as left right, and y starting 16 at top -16 at bottom
        self.rhoft = np.fft.rfft2(self.rho)

    def update_pot(self):
        self.pot = np.fft.irfft2(self.rhoft * self.kernelft)

    def update_forces(self):
        self.set_grad()
        self.f[:] = -self.grad

    def run(self, dt=1):
        self.x[:] = self.x[:] + dt * self.v
        self.update_rho()
        self.update_pot()
        self.update_forces()
        self.v[:]=self.v[:]+self.f*dt


if(__name__=="__main__"):
    obj = nbody(5000,128,-128,nside=128)

    # obj.ic_1part()
    # obj.ic_2part_circular()
    # obj.run(dt=0)
    # print(obj.grad)
    obj.ics_2gauss()
    # plt.ion()
    # plt.imshow(obj.pot)
    # plt.show()
    # plt.pause(100)

    frames = []  # for storing the generated images
    fig = plt.figure()
    for i in range(20000):
        obj.run(dt=0.01)
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
            # plt.clf()
            # plt.imshow(obj.rho, animated=True)
            frames.append([plt.imshow(obj.rho,animated=True)])
            if(len(frames)>50):
                break
            # plt.pause(0.05)

    ani = animation.ArtistAnimation(fig, frames, interval=200, blit=True,
                                    repeat_delay=500)
    ani.save('./5000_part_2gauss.gif', dpi=80, writer='imagemagick')
