import numpy as np
import matplotlib.pyplot as plt

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
        print(self.kernel)
        self.kernel[self.kernel <= self.soft ** 2] = self.soft ** 2
        self.kernel = (self.RES * self.RES * self.kernel) ** -0.5
        # print(self.kernel)
        self.kernelft = np.fft.rfft2(self.kernel)

    def ic_1part(self):
        # place one particle at (0,0) with zero velocity
        self.x[0,0] = 0
        self.x[0,1] = 0
        self.v[0,0] = 0
        self.v[0,1] = 0

    def update_rho(self):
        bins = self.RES*(np.arange(self.nside+1)-self.nside//2) #can equivalentlly use fftshift
        # since each cell is a bin, we have Nside bins, need Nside+1 points

        self.rho, xedges, yedges = np.histogram2d(self.x[:, 0], self.x[:, 1], bins=bins)
        self.rhoft = np.fft.rfft2(self.rho)

    def run(self):
        self.update_rho()


if(__name__=="__main__"):
    obj = nbody(1000,16,-16,nside=8)
    obj.run()
    plt.imshow(obj.rho)
    plt.show()
