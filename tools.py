import AO as tools
import numpy as np

#fourier filters

#create a line in an image given angle and distance from center
def line(angle, distance, size):
    x = np.linspace(-1,1,size)
    X,Y = np.meshgrid(x,x)
    return np.abs(X*np.cos(angle) + Y*np.sin(angle)) < distance
