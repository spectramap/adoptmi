import numpy as np
import matplotlib.pyplot as plt
import scipy 
import scipy.linalg as ln 
from scipy.optimize import curve_fit

from skimage.restoration import unwrap_phase

def wrap(phase):
    field = np.exp(1j*phase)
    return np.angle(field)

def bar(x, y, title = 'bar plot', xlabel = 'x', ylabel = 'y'):
    plt.figure(), plt.bar(x, y), plt.title(title), plt.xlabel(xlabel), plt.ylabel(ylabel)    
    
def crop_radial_image(phi, radios, offsetx=0, offsety=0):
    """narrow the pupil diameter of the DPP to the pupil diameter of the Raman microscope"""
    phi_new = np.zeros((2*radios, 2*radios))
    phi_new = phi[offsetx:int(2*radios+offsetx), offsety:int(2*radios+offsety)].copy()
    phi_new = phi_new*create_pupil(2*radios)
    return phi_new

#fourier transform
def ft(phi):
    """fourier transform"""
    phi_ft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(phi)))
    return phi_ft

def ift(phi_ft):
    """inverse fourier transform"""
    phi = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(phi_ft)))
    return phi

def gaussian(size, a=1, x0=0, y0=0, sigma_x=1, sigma_y=1, c=0):
    x = np.linspace(-1, 1, size)
    #y = np.linspace(-1, 1, image.shape[0])
    X, Y = np.meshgrid(x, x)
    exponent = np.exp(-((X-x0)**2/(2*sigma_x**2) + (Y-y0)**2/(2*sigma_y**2))) + c
    g = a * exponent/exponent.max()
    #return g.ravel()
    return g


def gaussian_function(size, a=1, x0=0, y0=0, sigma_x=1, sigma_y=1, c=0):
    x = np.linspace(0, size, size)
    #y = np.linspace(-1, 1, image.shape[0])
    X, Y = np.meshgrid(x, x)
    exponent = np.exp(-((X-x0)**2/(2*sigma_x**2) + (Y-y0)**2/(2*sigma_y**2))) + c
    g = a * exponent
    return g.ravel()
    #return g

def gaussian_fitting(image, plot = False, initial_guess = None):
    if initial_guess == None:
        initial_guess = [255, image.shape[0]/2, image.shape[0]/2, 1, 1, 0]
    x = np.linspace(-image.shape[0]/2, image.shape[0]/2, image.shape[0])
    X, Y = np.meshgrid(x, x)
    popt, pcov = curve_fit(gaussian_function, image.shape[0], image.ravel(), p0=initial_guess)
    if plot == True:
        show(gaussian_function(image.shape[0], *popt).reshape(image.shape), 'gaussian fitting')
    return popt

# create a pupil function
def create_pupil(size_mesh, centerx=0, centery=0, radius=0):
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
    mesh = np.zeros((size_mesh, size_mesh))
    if radius == 0:
        radius = size_mesh / 2.
    mesh = np.zeros((size_mesh, size_mesh))
    x = np.linspace(-size_mesh/2, size_mesh/2, size_mesh)
    y = np.linspace(-size_mesh/2, size_mesh/2, size_mesh)
    kx, ky = np.meshgrid(x, y)
    mesh[(kx-centerx)**2 + (ky-centery)**2 < radius**2] = 1
    return mesh

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
    coords = (np.arange(mesh_size) - mesh_size / 2. + 0.5) / (mesh_size / 2.)
    x, y = np.meshgrid(coords, coords)
    angular_matrix = np.arctan2(y, x)
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
    coords = (np.arange(mesh_size) - mesh_size / 2. + 0.5) / (mesh_size / 2.)
    x, y = np.meshgrid(coords, coords)
    radial_matrix = np.sqrt(x ** 2 + y ** 2)
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


def new_circle(radius, size, circle_centre=(0, 0), origin="middle"):
    """
    Create a 2-D array: elements equal 1 within a circle and 0 outside.

    The default centre of the coordinate system is in the middle of the array:
    circle_centre=(0,0), origin="middle"
    This means:
    if size is odd  : the centre is in the middle of the central pixel
    if size is even : centre is in the corner where the central 4 pixels meet

    origin = "corner" is used e.g. by psfAnalysis:radialAvg()

    Examples: ::

        circle(1,5) circle(0,5) circle(2,5) circle(0,4) circle(0.8,4) circle(2,4)
          00000       00000       00100       0000        0000          0110
          00100       00000       01110       0000        0110          1111
          01110       00100       11111       0000        0110          1111
          00100       00000       01110       0000        0000          0110
          00000       00000       00100

        circle(1,5,(0.5,0.5))   circle(1,4,(0.5,0.5))
           .-->+
           |  00000               0000
           |  00000               0010
          +V  00110               0111
              00110               0010
              00000

    Parameters:
        radius (float)       : radius of the circle
        size (int)           : size of the 2-D array in which the circle lies
        circle_centre (tuple): coords of the centre of the circle
        origin (str)  : where is the origin of the coordinate system
                               in which circle_centre is given;
                               allowed values: {"middle", "corner"}

    Returns:
        ndarray (float64) : the circle array
    """
    # (2) Generate the output array:
    C = np.zeros((size, size))

    # (3.a) Generate the 1-D coordinates of the pixel's centres:
    # coords = numpy.linspace(-size/2.,size/2.,size) # Wrong!!:
    # size = 5: coords = array([-2.5 , -1.25,  0.  ,  1.25,  2.5 ])
    # size = 6: coords = array([-3. , -1.8, -0.6,  0.6,  1.8,  3. ])
    # (2015 Mar 30; delete this comment after Dec 2015 at the latest.)

    # Before 2015 Apr 7 (delete 2015 Dec at the latest):
    # coords = numpy.arange(-size/2.+0.5, size/2.-0.4, 1.0)
    # size = 5: coords = array([-2., -1.,  0.,  1.,  2.])
    # size = 6: coords = array([-2.5, -1.5, -0.5,  0.5,  1.5,  2.5])

    coords = np.arange(0.5, size, 1.0)
    # size = 5: coords = [ 0.5  1.5  2.5  3.5  4.5]
    # size = 6: coords = [ 0.5  1.5  2.5  3.5  4.5  5.5]

    # (3.c) Generate the 2-D coordinates of the pixel's centres:
    x, y = np.meshgrid(coords, coords)

    # (3.d) Move the circle origin to the middle of the grid, if required:
    if origin == "middle":
        x -= size / 2.
        y -= size / 2.

    # (3.e) Move the circle centre to the alternative position, if provided:
    x -= circle_centre[0]
    y -= circle_centre[1]

    # (4) Calculate the output:
    # if distance(pixel's centre, circle_centre) <= radius:
    #     output = 1
    # else:
    #     output = 0
    mask = x * x + y * y <= radius * radius
    C[mask] = 1

    # (5) Return:
    return C

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
def zernike(n, m, r, theta, norm="Noll"):
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
    if norm != "Noll":
        return 0
    N = r.shape[0]
    zernike = np.zeros_like(r)
    if m > 0:
        for s in range(int((n - m) / 2) + 1):
            zernike += ((-1) ** s * scipy.special.binom(n - s, s) * scipy.special.binom(n - 2 * s, (n - m) / 2 - s) * r ** (n - 2 * s) * np.cos(m * theta))
        zernike = zernike*np.sqrt(2*(n+1))
    elif m < 0:
        for s in range(int((n - abs(m)) / 2) + 1):
            zernike += ((-1) ** s * scipy.special.binom(n - s, s) * scipy.special.binom(n - 2 * s, (n - abs(m)) / 2 - s) * r ** (n - 2 * s) * np.sin(np.abs(m) * theta ))
        zernike = zernike*np.sqrt(2*(n+1))
    else:
        for s in range(int(n/2 + 1)):
            zernike += ((-1) ** s * scipy.special.binom(n - s, s) * scipy.special.binom(n - 2 * s, n / 2 - s) * r ** (n - 2 * s))
        zernike = zernike*np.sqrt(n+1)
    return zernike*np.less_equal(r, 1.0)*new_circle(N/2., N)

def unwrap(phase):
    return unwrap_phase(phase)
    
def zernike_index(order, size, index="OSA", norm="Noll"):
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
    #rho = r/r.max()
    zern = zernike(n, m, r, angular)
    pupil = create_pupil(size)
    if norm == "Noll":
        return zern
    elif norm == "rms":
        zern[pupil] /= np.sqrt(np.sum(zern[pupil]**2)/len(zern[pupil]))
        return zern
    elif norm == "unit":
        zern[pupil] -= zern[pupil].min()
        zern[pupil] /= zern[pupil].max()
        return zern
    elif norm == "unit2":
        zern[pupil] -= zern[pupil].min()
        zern[pupil] /= zern[pupil].max()*2-1
        return zern

def show(phase, ybarlabel="Intensity", cmap="jet", xlabel="x", ylabel="y", title="Phase", path=None):
    if np.iscomplexobj(phase):
        fig, ax = plt.subplots(1, 2)
        im1 = ax[0].imshow(np.abs(phase), cmap=cmap)
        ax[0].set_xlabel(xlabel)
        ax[0].set_ylabel(ylabel)
        ax[0].set_title("Amplitude")
        c1bar = fig.colorbar(im1, location="bottom")
        c1bar.set_label("Intensity")
        im2 = ax[1].imshow(np.angle(phase), cmap=cmap)
        ax[1].set_xlabel(xlabel)
        ax[1].set_ylabel(ylabel)
        ax[1].set_title("Phase")
        c2bar = fig.colorbar(im2, location="bottom")
        c2bar.set_label("Phase [rad]")
    else:
        fig, ax = plt.subplots()
        im = ax.imshow(phase, cmap = cmap)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        cbar = fig.colorbar(im)
        cbar.set_label(ybarlabel)
    if path != None:
        fig.savefig(path+"/"+title+".png", dpi=300)
            
def zernike_decompose(phase, J:list, index="OSA", plot = False, norm = "Noll"):
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
        G[j,:] = zernike_index(J[j], phase.shape[0], index=index, norm=norm)[pupil]
    
    phase_flat = phase[pupil]
    inv = ln.inv(np.dot(G, G.T))
    coeff = np.dot(inv, G)
    coefficients = np.dot(coeff, phase_flat)

    if plot == True:
        plt.figure(), plt.bar(J, coefficients, width=0.8)
        plt.xlabel(index), plt.ylabel(norm)
    return coefficients

def zernike_multi(orders, coefficients, N, index="OSA", norm = "Noll", plot = False):
    out_arr = np.zeros((N, N))
    for count in range(len(orders)):
        phase = zernike_index(orders[count], N, index, norm)
        out_arr += coefficients[count]*phase
    if plot == True:
        plt.figure(), plt.imshow(out_arr), plt.colorbar()
    return out_arr
    
#fourier filters

#create a line in an image given angle and distance from center
def line(angle, distance, size, num_lines=1, angular_width = 0):
    #angle in radians
    #distance in pixels
    #size of image
    #num_lines is the number of lines to create
    #angular_width is the angular width of the line
    #convert degrees to radians
    angle = angle*np.pi/180
    angular_width = angular_width*np.pi/180

    x = np.linspace(-size/2,size/2,size)
    X,Y = np.meshgrid(x,x)
    tmp = np.zeros((size,size))
    for i in range(num_lines):
        tmp+=np.abs(X*np.cos(angle+angular_width/num_lines*i) + Y*np.sin(angle+angular_width/num_lines*i)) < distance
    return tmp == 0