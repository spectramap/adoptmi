from tools import*
    
class AO:
    def __init__(self, name, wave, D, size_mesh):
        self.name = name
        self.lambda_ = wave
        self.D = D
        self.size_mesh = size_mesh
        x = np.linspace(-D/2, D/2, size_mesh)
        y = np.linspace(-D/2, D/2, size_mesh)
        kx, ky = np.meshgrid(x, y)
        self.beam = kx**2 + ky**2 < (D/2)**2
        self.phase = np.zeros((size_mesh, size_mesh))

    def get_beam(self):
        return self.beam
    
    def decompose(self, coefficients:list, index="OSA"):
        return zernike_decompose(self, coefficients, index=index)

    def set_pupil(self, D):
        pupil = create_pupil(D)
        self.beam *= pupil
        self.phase *= pupil

    #read phase image
    def read_phase(self, path):
        self.phase = plt.imread(path)
    
    #create a zernike polynomial
    def zernike(self, orders:list, coefficients:list, index="OSA"):
        self.phase = zernike_multi(orders, coefficients, self.size_mesh, index=index)

    #show both phase and beam the same figure
    def show(self):
        plt.subplot(1, 2, 1)
        im1 = plt.imshow(self.beam)
        plt.colorbar(im1)
        plt.subplot(1, 2, 2)
        im2 = plt.imshow(self.phase)
        plt.colorbar(im2)

    def show_beam(self):
        plt.imshow(self.beam)
        plt.colorbar()

    def show_phase(self):
        plt.imshow(self.phase)
        plt.colorbar()

    def set_phase(self, phase):
        self.phase = phase
    
    def get_phase(self):
        return self.phase
    
    def set_beam(self, beam):
        self.beam = beam
    
    def fft(self):
        #fourier transform of an image using numpy and considering shifts
        beam = self.beam*np.exp(self.phase*1j)
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(beam)))
    
    def ifft(self):
        #inverse fourier transform of an image using numpy and considering shifts
        beam = self.beam*np.exp(self.phase*1j)
        return ift(beam)

    def wrap(self):
        self.phase = np.angle(np.exp(1j*self.phase))

    def unwrap(self, kind='fast'):
        if kind == 'fast':
            self.phase = np.unwrap(self.phase, axis = -1)
        elif kind == 'robust':
            self.phase = unwrap_phase(self.phase)


