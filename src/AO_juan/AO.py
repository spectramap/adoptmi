import numpy as np
import matplotlib.pyplot as plt
import scipy 
import scipy.linalg as ln 
from skimage.restoration import unwrap_phase
import aotools as ao
from scipy.optimize import least_squares

def crop_radial_image(phi, D):
    """narrow the pupil diameter of the DPP to the pupil diameter of the Raman microscope"""
    phi_new = np.zeros((D, D))
    phi_new = phi[int((phi.shape[0]-D)/2):int((phi.shape[0]+D)/2), int((phi.shape[1]-D)/2):int((phi.shape[1]+D)/2)]
    phi_new = phi_new*create_pupil(D, D)
    return phi_new

# create a pupil function
def create_pupil(size_mesh):
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
    x = np.linspace(-size_mesh/2, size_mesh/2, size_mesh)
    y = np.linspace(-size_mesh/2, size_mesh/2, size_mesh)
    kx, ky = np.meshgrid(x, y)
    return kx**2 + ky**2 < (size_mesh/2)**2

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
    
def norm(image):
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
    domain = create_pupil(image.shape[0])
    extended_image = image.copy()*domain
    extended_image[domain] = extended_image[domain] - extended_image[domain].min()
    extended_image[domain] = extended_image[domain]/extended_image[domain].max()*2 - 1
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

def unwrap(phase):
    return norm(unwrap_phase(phase))*np.pi*create_pupil(phase.shape[0])
    
def zernike_index(order, size, index="OSA"):
    """
    Generate a circular limited zernike polynomial.
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

def show(phase):
    plt.figure()
    if type(phase) == complex:
        plt.subplot(1, 2, 1)
        im1 = plt.imshow(phase)
        plt.colorbar(im1)
        plt.subplot(1, 2, 2)
        im2 = plt.imshow(phase)
        plt.colorbar(im2)
    else:
        plt.imshow(phase)
        plt.colorbar()
        
def zernikes(phase, J:list, index="OSA", plot = False):
    """
    Decompose an image into zernike coefficients.
    Parameters
    ----------
    beam : array
        phase to decompose.
    J : list
        zernike coefficients.
    index : str
        Indexing of the zernike polynomials.
    Returns
    -------
    zernike : list
        Zernike coefficients.
    """
    pupil = create_pupil(phase.shape[0])
    s = pupil[pupil[:,:] == True].shape[0]    
    G = np.zeros((len(J), s))
    
    #phase_zero = phase - 2
    for j in range(len(J)):
        #G[j,:] = zernike_index(J[j], unwrapped_phi.shape[0], index='OSA')[pupil]
        G[j,:] = ao.zernike_noll(J[j], phase.shape[0])[pupil]
    
    phase_flat = phase[pupil]
    inv = ln.inv(np.dot(G, G.T))
    coeff = np.dot(inv, G)
    coefficients = np.dot(coeff, phase_flat)

    if plot == True:
        plt.figure(), plt.bar(J, coefficients, width=0.8)
        plt.xlabel(index)
    return coefficients

def zernike_multi(orders, coefficients, N, index="OSA"):
    out_arr = np.zeros((N, N))
    for count in range(len(orders)):
        phase = zernike_index(orders[count], N, index)
        phase = norm(phase)
        out_arr += coefficients[count]*phase
    return out_arr*create_pupil(N)
    
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


reference = ao.zernike_noll(4, 64)
manual = zernike_index(4, 64)
