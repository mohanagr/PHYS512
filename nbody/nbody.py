import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import time

np.random.seed(42)

@nb.njit(parallel=True)
def get_grad(x,pot,RES,grad):
    nside = pot.shape[0]
    npart = x.shape[0]
    for i in nb.prange(npart):
        # print("Particle positions:", x[i])
        irow = int(nside /2 - x[i, 1]/RES)  # row num is given by y position up down
        icol = int(nside /2 + x[i, 0]/RES)  # col num is given by x position left right
        grad[i, 1] = -0.5 * (pot[(irow - 1), icol] - pot[(irow + 1) % nside, icol]) / RES
        # y decreases with high row num.
        grad[i, 0] = -0.5 * (pot[irow, (icol + 1) % nside] - pot[irow, icol - 1]) / RES
        # print(self.grad)

@nb.njit(parallel=True)
def hist2d(x, mat, RES):
    # this is better than numpy.hist because of parallelization
    nside = mat.shape[0]

    for i in nb.prange(x.shape[0]):
        irow = int(nside/2 - x[i, 1]/RES)
        icol = int(nside/2 + x[i, 0]/RES)
        mat[irow,icol]+=1

# def enforce_period(x):
#     XMAX=128
#     XMIN=-128
#     x[np.logical_or(x > XMAX, x < XMIN)] = x[np.logical_or(x > XMAX, x < XMIN)] % (XMAX - XMIN)
#     x[x > XMAX] = x[x > XMAX] - (XMAX - XMIN)

# @nb.njit(parallel=True)
# def enforce_period_nb(x):
#     XMAX=128
#     XMIN=-128
#     for i in range(x.shape[0]):
#         if(x[i,0]>XMAX or x[i,0]<XMIN):
#             x[i,0]=x[i,0]%(XMAX-XMIN)
#             if(x[i,0]>XMAX): x[i,0]=x[i,0]-XMAX+XMIN
#         if(x[i,1]>XMAX or x[i,1]<XMIN):
#             x[i,1]=x[i,1]%(XMAX-XMIN)
#             if (x[i, 1] > XMAX): x[i, 1] = x[i, 1] - XMAX + XMIN

@nb.njit
def enforce_period_nb(x,XMAX,XMIN):
    W = XMAX-XMIN
    for i in range(x.shape[0]):   #prange gives an error here. some bug in numba on googling
        while(x[i,0]>XMAX):
            x[i,0]-=W
        while(x[i,0]<XMIN):
            x[i,0]+=W
        while(x[i,1]>XMAX):
            x[i,1]-=W
        while(x[i,1]<XMIN):
            x[i,1]+=W

# def enforce_period(x):
#     W = 256
#     XMAX = 128
#     XMIN = -128
#     x[x>XMAX] = x[x>XMAX] - W
#     x[x<XMIN] = x[x<XMIN] + W

class nbody():

    def __init__(self, npart, xmax, xmin, nside=1024,soft=1):
        self.nside=nside
        self.x=np.zeros([npart,2])
        self.f=np.zeros([npart,2])
        self.grad=np.zeros([npart,2])
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

        self.v[0, 1] = 0.2
        self.v[1, 1] = 0.2


    def ic_gauss(self):
        self.x[:] = np.random.randn(self.npart, 2) * self.XMAX / 4  # so that 95% within XMIN - XMAX

    def ic_2gauss(self, vel):
        self.x[:]=np.random.randn(self.npart,2)*self.XMAX/4
        self.x[:self.npart//2,1]=self.x[:self.npart//2,1]-self.XMAX/2.4
        self.x[self.npart//2:,1]=self.x[self.npart//2:,1]+self.XMAX/2.4
        self.v[:self.npart//2,0]= vel # 3 for 5k
        self.v[self.npart//2:,0]=-vel

    def ic_3part(self):
        self.x[0,:] = [0,2]
        self.x[1,:] = [-2, -2]
        self.x[2,:] = [2, -2]


    def update_rho(self):
        enforce_period_nb(self.x,self.XMAX,self.XMIN)
        # self.rho, xedges, yedges = np.histogram2d(x[:, 1], x[:, 0], bins=bins)

        self.rho[:] = 0
        hist2d(self.x,self.rho,self.RES)

        # we want y as top down, self.x as left right, and y starting 16 at top -16 at bottom
        self.rhoft = np.fft.rfft2(self.rho)

    def update_pot(self):
        self.update_rho()
        self.pot = np.fft.irfft2(self.rhoft * self.kernelft, [self.nside, self.nside])

    def update_forces(self):
        get_grad(self.x, self.pot, self.RES,self.grad)
        self.f[:] = self.grad
        # return get_grad(self.x, self.pot, self.RES)

    def run_leapfrog(self, dt=1, verbose=False):
        self.x[:] = self.x[:] + dt * self.v
        self.update_pot()
        self.update_forces()
        self.v[:]=self.v[:]+self.f*dt


    def run_leapfrog2(self, dt=1, verbose=False):
        #initialize cur_f to use this version
        self.x[:] = self.x[:] + dt * self.v[:] + self.f * dt * dt /2
        self.update_pot()
        get_grad(self.x, self.pot, self.RES,self.grad)
        self.v[:] = self.v[:] + (self.f + self.grad)*dt/2
        self.f[:] = self.grad

def init_anim():
    img.set_data(np.zeros((1024,1024)))
    return img,

def animate(i):

    #PE = 0.5* np.sum(obj.pot * obj.rho) # USE THIS WHEN USING LEAPFROG2. comment PE1 and PE2. TE = KE+PE

    PE1 = 0.25 * np.sum(obj.pot * obj.rho)  # at 0 step
    KE=0.5*np.sum(obj.v**2) # at half step
    st = time()
    obj.run_leapfrog(dt=0.001)
    et = time()
    PE2 = 0.25 * np.sum(obj.pot * obj.rho)  # at 1 step
    if(i%10==0):
        print("FRAME:",i, et - st, PE1 + PE2 + KE)
    img.set_data(obj.rho**0.5)

    return img,


if(__name__=="__main__"):
    NSIDE = 1024
    XMAX=128
    XMIN=-128
    RES=(XMAX-XMIN)/NSIDE
    NPART = 1000000
    SOFT = 2
    TSTEP = SOFT * RES / np.sqrt(NPART)  # this is generally smaller than eps**3/2
    obj = nbody(NPART,XMAX,XMIN,soft=SOFT,nside=NSIDE)


    # obj.ic_1part()
    # obj.ic_2part_circular()

    obj.ic_2gauss(50) #to use this change lims to +/- 128 and particles to 5000
    # obj.ic_3part()
    # obj.ic_gauss()
    obj.update_pot()
    # obj.update_forces() # use this only when using leapfrog2, and change Energy condition in animate

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches((6,6))
    img = ax.imshow(obj.rho ** 0.5,cmap='inferno',animated=True)
    no_labels = 9  # how many labels to see on axis x
    step = int(NSIDE / (no_labels - 1))  # step between consecutive labels
    positions = np.arange(0, NSIDE + 1, step)  # pixel count at label position
    labels = -positions * RES + XMAX  # labels you want to see --- imshow plots origin at top
    ax.set_yticks(positions, labels)
    ax.set_xticks(positions, labels)

    FFwriter = animation.FFMpegWriter(fps=40, extra_args=['-vcodec', 'libx264'])
    try:
        anim = animation.FuncAnimation(fig, animate, init_func=init_anim, repeat=True,frames=4000, interval=20, blit=True, repeat_delay=1000)
    except KeyboardInterrupt as e:
        pass
    finally:
        anim.save('./dump.mp4',writer=FFwriter)





