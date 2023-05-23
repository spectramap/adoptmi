import numpy as np
import matplotlib.pyplot as plt
import scipy 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from scipy.linalg import lstsq
from skimage.restoration import unwrap_phase


def crop_phases_to_smaller_pupil(pupil_old, pupil_new, phi):
    """narrow the pupil diameter of the DPP to the pupil diameter of the Raman microscope"""
    ux = pupil_old/phi.shape[1] #pixel size 
    delta = (pupil_old - pupil_new)/2/ux
    phi_new = phi[int(np.round(delta)):-int(np.round(delta)),int(np.round(delta)):-int(np.round(delta))]
    pupil_Olympus = pupil(phi_new.shape[1], phi_new.shape[1])
    phi_new *= pupil_Olympus
    #MW = sum(phi_new, axis = (1,2))/sum(pupil_Olympus) #removoing mean values
    #phi_new -= MW[:,newaxis,newaxis]
    return phi_new

# convert window path to python path
def path_convert(path):
    path = path.replace('\\', '/')
    return path

# create a pupil function
def create_pupil(size_mesh, D):
    """
    Create a pupil function.
    Parameters
    ----------
    mesh_size : int
        Size of the pupil function.
    radius : float
        Radius of the pupil function.
    phase : 2D array
        Phase of the pupil function.
    phase_mask : 2D array
        Mask for the phase of the pupil function.
    Returns
    -------
    pupil_function : 2D array
        Pupil function.
    """
    x = np.linspace(-D/2, D/2, size_mesh)
    y = np.linspace(-D/2, D/2, size_mesh)
    kx, ky = np.meshgrid(x, y)
    return kx**2 + ky**2 < (D/2)**2

#unwrap phase from an image
def unwrap_phase(image):
    """
    Unwrap phase from an image.
    Parameters
    ----------
    image : 2D array
        Image with wrapped phase.
    Returns
    -------
    unwrapped_phase : 2D array
        Image with unwrapped phase.
    """
    unwrapped_phase = np.unwrap(np.unwrap(image, axis=0), axis=1)
    return unwrapped_phase

#create a 2d angular matrix
def angular_matrix(mesh_size):
    """
    Create a 2D angular coordinate matrix.
    Parameters
    ----------
    size : int
        Size of the matrix.
    Returns
    -------
    angular_matrix : 2D array
        2D angular coordinate matrix.
    """
    y, x = np.indices((mesh_size, mesh_size))
    angular_matrix = np.arctan2(x - mesh_size / 2, y - mesh_size / 2)
    angular_matrix = np.where(angular_matrix < 0, angular_matrix + 2 * np.pi, angular_matrix)
    return angular_matrix

# create a 2d radial matrix
def radial_matrix(mesh_size):
    """
    Create a 2D radial matrix.
    Parameters
    ----------
    size : int
        Size of the matrix.
    Returns
    -------
    radial_matrix : 2D array
        2D radial matrix.
    """
    y, x = np.indices((mesh_size, mesh_size))
    radial_matrix = np.sqrt((x - mesh_size / 2) ** 2 + (y - mesh_size / 2) ** 2)
    return radial_matrix

# get noll index from m and n
def index2noll(n, m):
    """
    Get noll index from m and n.
    Parameters
    ----------
    m : int
        Zernike m index.
    n : int
        Zernike n index.
    Returns
    -------
    noll : int
        Noll index.
    """
    add_n = 1
    if m > 0:
        if (n % 4) == 0:
            add_n = 0
        elif ((n - 1) % 4) == 0:
            add_n = 0
    elif m < 0:
        if ((n - 2) % 4) == 0:
            add_n = 0
        elif ((n - 3) % 4) == 0:
            add_n = 0
    return (n*(n + 1))//2 + abs(m) + add_n

def index2osa(n,m):
    """
    Get OSA index from m and n.
    Parameters
    ----------
    m : int
        Zernike m index.
    n : int
        Zernike n index.
    Returns
    -------
    osa : int
        OSA index.
    """
    return((n*(n+2) + m)/2)

def noll2index(noll_index):

    """
    Convert noll index to osa index right way.
    Parameters
    ----------
    noll : int
        Noll index.
    Returns
    -------
    osa : int
        OSA index.
    """
    n = np.floor(np.sqrt(2*noll_index - 1) - 0.5)
    if n%2 == 0:
        m = 2*np.floor((2*noll_index + 1 - n*(n+1))/4)
    else:
        m = 2*np.floor((2*(noll_index + 1) - n*(n+1))/4) - 1
    
    if noll_index%2 != 0:
        m = -m
    return n, m

def osa2index(osa_index):
    """
    Convert osa index to noll index right way.
    Parameters
    ----------
    osa : int
        OSA index.
    Returns
    -------
    noll : int
        Noll index.
    """
    n = np.ceil((-3+np.sqrt(9+8*osa_index))/2)
    m = 2*osa_index - n*(n+2)
    return n, m

def noll2osa(noll_index):
    """
    Convert noll index to osa index right way.
    Parameters
    ----------
    noll : int
        Noll index.
    Returns
    -------
    osa : int
        OSA index.
    """
    n, m = noll2index(noll_index)
    return index2osa(n, m)

def osa2noll(osa_index):
    """
    Convert osa index to noll index right way.
    Parameters
    ----------
    osa : int
        OSA index.
    Returns
    -------
    noll : int
        Noll index.
    """
    n, m = osa2index(osa_index)
    return index2noll(n, m)

def norm(image, upper, lower):
    """
    Extend the domain of an image.
    Parameters
    ----------
    image : 2D array
        Image to extend.
    range : int
        Number of pixels to extend the image.
    Returns
    -------
    extended_image : 2D array
        Image with extended domain.
    """
    extended_image = image.copy()
    tmp = extended_image - extended_image.min()
    extended_image = tmp/tmp.max()*2 - 1
    return extended_image

# create an unit circle zernike polynomial
def zernike(n, m, r, theta, mask=None):
    """
    Create a normalized Zernike polynomials.
    Parameters
    ----------
    n : int
        Order of the polynomial.
    m : int
        Repetition of the polynomial.
    r : 2D array
        Radial coordinate matrix.
    theta : 2D array
        Angular coordinate matrix.
    mask : 2D array, optional
        Mask of the image. The pixels with value 0 will be ignored.
    Returns
    -------
    zernike : 2D array
        Zernike polynomial.
    """
    if mask is None:
        mask = np.ones_like(r)
    zernike = np.zeros_like(r)
    if m > 0:
        for s in range(int((n - abs(m)) / 2) + 1):
            zernike += ((-1) ** s * scipy.special.binom(n - s, s) * scipy.special.binom(n - 2 * s, (n - abs(m)) / 2 - s) * r ** (n - 2 * s) * np.cos(m * theta) * mask)
        zernike = zernike*np.sqrt(2*(n+1))
    elif m < 0:
        for s in range(int((n - abs(m)) / 2) + 1):
            zernike += ((-1) ** s * scipy.special.binom(n - s, s) * scipy.special.binom(n - 2 * s, (n - abs(m)) / 2 - s) * r ** (n - 2 * s) * np.sin(np.abs(m) * theta * mask) * mask)
        zernike = zernike*np.sqrt(2*(n+1))
    else:
        for s in range(int(n/2 + 1)):
            zernike += ((-1) ** s * scipy.special.binom(n - s, s) * scipy.special.binom(n - 2 * s, n / 2 - s) * r ** (n - 2 * s) * mask)
        zernike = zernike*np.sqrt(n+1)
    return zernike

def zernike_index(order, size, index="OSA"):
    """
    Generate a zernike polynomial.
    Parameters
    ----------
    order : int
        Zernike order.
    size : int
        Size of the array.
    index : str
        Indexing of the zernike polynomials.
    Returns
    -------
    zernike : 2D array
        Zernike polynomial.
    """
    if index == "OSA":
        n, m = osa2index(order)
    elif index == "Noll":
        n, m = noll2index(order)
    else:
        raise ValueError("Indexing must be either OSA or Noll")

    #r is a radial matrix
    r = radial_matrix(size)
    angular = angular_matrix(size)
    rho = r/r.max()
    return zernike(n, m, rho, angular)

def zernikes(beam, J, index="OSA"):
    """
    Decompose an image into zernike coefficients.
    Parameters
    ----------
    beam : object
        phase to decompose.
    J : int
        zernike coefficients.
    index : str
        Indexing of the zernike polynomials.
    Returns
    -------
    zernike : list
        Zernike coefficients.
    """
    s = beam.phase.shape[0]
    G = np.zeros((len(J), s*s))
    D = np.zeros((len(J), s, s))
    for j in range(len(J)):
        D[j,:,:] = norm(zernike_index(J[j], s, index=index), 1, 1) * beam.beam
        for j in range(len(J)):
            G[j,:] = D[j,:,:].flatten()
    phase_flat = beam.phase.flatten()
    coefficients, _, _, _ = lstsq(G.T, phase_flat)
    return coefficients

def zernike_multi(orders, coefficients, N, index="OSA"):
    out_arr = np.zeros((N, N))
    for count in range(len(orders)):
        phase = zernike_index(orders[count], N, index)
        phase = norm(phase, 1, -1)
        out_arr += coefficients[count]*phase
    return out_arr
    
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
    
    def zernikes(self, coefficients:list, index="OSA"):
        return zernikes(self, coefficients, index=index)

    def set_pupil(self, D):
        x = np.linspace(-D/2, D/2, self.size_mesh)
        y = np.linspace(-D/2, D/2, self.size_mesh)
        X, Y = np.meshgrid(x, y)
        pupil = X**2 + Y**2 < (D/2)**2
        self.beam *= pupil
        self.phase *= pupil

    #read phase image
    def read_phase(self, path):
        self.phase = plt.imread(path)
    
    #create a zernike polynomial
    def zernike(self, orders:list, coefficients:list, index="OSA"):
        out_arr = np.zeros((self.size_mesh, self.size_mesh))
        for count in range(len(orders)):
            phase = zernike_index(orders[count], self.size_mesh, index)
            phase = norm(phase, 1, -1)
            out_arr += coefficients[count]*phase
        self.phase = out_arr
        self.set_pupil(self.D)

    #show both phase and beam the same figure
    def show(self):
        plt.subplot(1, 2, 1)
        im1 = plt.imshow(self.beam)
        plt.colorbar(im1)
        plt.subplot(1, 2, 2)
        im2 = plt.imshow(self.phase)
        plt.colorbar(im2)
        plt.show()

    def show_beam(self):
        plt.imshow(self.beam)
        plt.colorbar()
        plt.show()

    def show_phase(self):
        plt.imshow(self.phase)
        plt.colorbar()
        plt.show()

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
        return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(beam)))

    def wrap(self):
        self.phase = np.angle(np.exp(1j*self.phase))

    def unwrap(self, kind='fast'):
        if kind == 'fast':
            self.phase = np.unwrap(self.phase, axis = -1)
        elif kind == 'robust':
            self.phase = unwrap_phase(self.phase)

beam = AO('AO1', 1.5e-6, 1.3, 256)
beam.zernike([4, 3, 5], [4, 0.3, 0.9], index="OSA")