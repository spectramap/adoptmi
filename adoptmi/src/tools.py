import numpy as np
import matplotlib.pyplot as plt
import scipy 
import scipy.linalg as ln 
from skimage.restoration import unwrap_phase

#def crop a circle region of interest from an image
def crop_circle(imarray, center, radius):
    x, y = center
    crop = np.zeros((2*radius, 2*radius))
    crop = imarray[x-radius:x+radius, y-radius:y+radius]
    return crop*ao.create_pupil(2*radius)

def high_pass_filter(imarray, radius):
    x, y = imarray.shape
    filter = np.zeros((x, y))
    for i in range(x):
        for j in range(y):
            if np.sqrt((i-x/2)**2 + (j-y/2)**2) > radius:
                filter[i,j] = 1
    return filter

def low_pass_filter(imarray, radius):
    x, y = imarray.shape
    filter = np.zeros((x, y))
    for i in range(x):
        for j in range(y):
            if np.sqrt((i-x/2)**2 + (j-y/2)**2) < radius:
                filter[i,j] = 1
    return filter   

#create a bandwidth filter
def annular_filter(imarray, inner_radius, outer_radius):
    x, y = imarray.shape
    filter = np.zeros((x, y))
    for i in range(x):
        for j in range(y):
            if np.sqrt((i-x/2)**2 + (j-y/2)**2) < outer_radius and np.sqrt((i-x/2)**2 + (j-y/2)**2) > inner_radius:
                filter[i,j] = 1
    return filter
    
#gaussian filter
def gaussian(size_mesh, sigma=1):
    #define beam in the pupil 
    x = np.linspace(-1,1,size_mesh)
    y = np.linspace(-1,1,size_mesh)
    X,Y = np.meshgrid(x,y)
    R = np.sqrt(X**2+Y**2)
    return np.exp(-(X**2+Y**2)/sigma**2) #beam profile

#fourier transform of an image
def ft(image):
    """
    Calculate the Fourier transform of an image.
    Parameters
    ----------
    image : 2D array
        Image.
    Returns
    -------
    image_ft : 2D array
        Fourier transform of the image.
    """
    image_ft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image)))
    return image_ft

#inverse fourier transform of an image
def ift(image_ft):
    """
    Calculate the inverse Fourier transform of an image.
    Parameters
    ----------
    image_ft : 2D array
        Fourier transform of the image.
    Returns
    -------
    image : 2D array
        Image.
    """
    image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(image_ft)))
    return image

def bar(x, y, name = 'bar plot', xlabel = 'x', ylabel = 'y'):
    plt.figure(), plt.bar(x, y), plt.title(name), plt.xlabel(xlabel), plt.ylabel(ylabel)    

def crop_circle(imarray, center, radius):
    x, y = center
    crop = np.zeros((2*radius, 2*radius))
    crop = imarray[x-radius:x+radius, y-radius:y+radius]
    return crop*create_pupil(2*radius)

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

    # (3.b) Just an internal sanity check:
    if len(coords) != size:
        raise exception.Bug("len(coords) = {0}, ".format(len(coords)) +
                             "size = {0}. They must be equal.".format(size) +
                             "\n           Debug the line \"coords = ...\".")

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
    
def zernike_index(order, size, index="Noll", norm="Noll"):
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
    zern = zernike(n, m, r, angular)
    if norm == "Unit":
        tmp = zern - zern.min()
        zern = tmp/tmp.max()*2 - 1
        return zern
    elif norm == "Noll":
        return zern

def show(beam, cmap="jet", xlabel="x", ylabel="y"):
    plt.figure()
    if np.sum(np.imag(beam)) != 0:
        plt.subplot(1, 2, 1)
        im1 = plt.imshow(np.abs(beam), cmap=cmap)
        plt.colorbar(im1)
        plt.subplot(1, 2, 2)
        im2 = plt.imshow(np.angle(beam), cmap=cmap)
        plt.colorbar(im2)
    else:
        plt.imshow(beam, cmap=cmap)
        plt.colorbar()
        
def zernike_decompose(phase, J:list, index="Noll", plot = False, norm = "Noll"):
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
    if norm != "Noll":
        return 0
    pupil = create_pupil(phase.shape[0])
    s = pupil[pupil[:,:] == True].shape[0]    
    G = np.zeros((len(J), s))
    
    #phase_zero = phase - 2
    for j in range(len(J)):
        G[j,:] = zernike_index(J[j], unwrapped_phi.shape[0], index='Noll')[pupil]
        #G[j,:] = ao.zernike_noll(J[j], phase.shape[0])[pupil]
    
    phase_flat = phase[pupil]
    inv = ln.inv(np.dot(G, G.T))
    coeff = np.dot(inv, G)
    coefficients = np.dot(coeff, phase_flat)

    if plot == True:
        plt.figure(), plt.bar(J, coefficients, width=0.8)
        plt.xlabel(index), plt.ylabel(norm)
    return coefficients

def zernike_multi(orders, coefficients, N, index="Noll", norm = "Noll", plot = False):
    out_arr = np.zeros((N, N))
    for count in range(len(orders)):
        phase = zernike_index(orders[count], N, index, norm)
        out_arr += coefficients[count]*phase
    if plot == True:
        plt.figure(), plt.imshow(out_arr), plt.colorbar()
    return out_arr
    
#fourier filters

#create a line in an image given angle and distance from center
def line(angle, distance, size):
    x = np.linspace(-1,1,size)
    X,Y = np.meshgrid(x,x)
    return np.abs(X*np.cos(angle) + Y*np.sin(angle)) < distance
