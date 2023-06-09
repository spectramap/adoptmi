#%%
import tools as ao
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from PIL import Image
import os


#%% Paramerters

path_images = "C:\\Users\\q125jm\\Documents\\Raman\\DPP calibration\\"
num_act = 63
radius = 32
num_steps = 11
voltages = np.linspace(0, 300, num_steps)

#%% testing one image for the phase processing
im1 = np.array(Image.open(path_images+'_4_120.0_1.tiff'))
im2 = np.array(Image.open(path_images+'_4_120.0_2.tiff'))
im3 = np.array(Image.open(path_images+'_4_120.0_3.tiff'))
imarray = (im1+im2+im3)/3

cropped = ao.crop_radial_image(imarray, radios=470, offsetx=152, offsety=214)
ao.show(cropped)
ft = ao.ft(cropped)

ao.show(np.log(ft))

cropped_ft = ao.crop_radial_image(ft, radios=32, offsetx=471, offsety=470+189)

ao.show(np.log(cropped_ft))

ift_cropped = np.angle(ao.ift(cropped_ft)) * ao.create_pupil(32*2)
ao.show(ift_cropped)

# %% phase retrieval
radius = 24
num_steps = 11
voltages = np.linspace(0, 300, num_steps)
phase_control = np.zeros((num_act, num_steps, 2*radius, 2*radius))

for i in range(num_act):
    print('Actuator: ', i)
    im1 = np.array(Image.open(path_images+'_'+str(i+1)+'_'+str(150.0)+'_1.tiff'))
    im2 = np.array(Image.open(path_images+'_'+str(i+1)+'_'+str(150.0)+'_2.tiff'))
    im3 = np.array(Image.open(path_images+'_'+str(i+1)+'_'+str(150.0)+'_3.tiff'))

    offset = (im1+im2+im3)/3

    count = 0
    for voltage in voltages:
        im1 = np.array(Image.open(path_images+'_'+str(i+1)+'_'+str(voltage)+'_1.tiff'))
        im2 = np.array(Image.open(path_images+'_'+str(i+1)+'_'+str(voltage)+'_2.tiff'))
        im3 = np.array(Image.open(path_images+'_'+str(i+1)+'_'+str(voltage)+'_3.tiff'))

        img = (im1+im2+im3)/3 - offset

        img_cropped = ao.crop_circle(img, (624, 682), 470)
        #ao.show(offset_cropped)
        img_ft = ao.ft(img_cropped)
        #ao.show(np.log(offset_ft))

        img_cropped_ft = ao.crop_circle(img_ft, (471, 470+189), radius)

        #ao.show(np.log(offset_cropped_ft))

        img_ift_cropped = np.angle(ao.ift(img_cropped_ft))*ao.create_pupil(radius*2)
        #ao.show(img_ift_cropped), plt.title('offset '+str(i+1))  
        phase_control[i, count, :, :] = img_ift_cropped
        count += 1
#save data
np.save('phase_control.npy', phase_control)

# %%read data and continue processing

phase_control = np.load('phase_control.npy')

size_mesh = phase_control.shape[2]

pupil = ao.create_pupil(size_mesh)
num_pupil = len(pupil[pupil])

#removing mean value of phase control
M = phase_control.copy()
for m in range(num_act):
    for i in range(num_steps):
        M[m, i, pupil] -= M[m, 5, pupil] #subtracting the center point of the phase control
        M[m, i, :,:] = ao.unwrap(M[m, i, :,:])*ao.create_pupil(size_mesh)
        M[m, i, pupil] -= np.mean(M[m, i, pupil]) #subtracting the mean value of the influence functions
        #M[m, i, pupil] -= M[m, i, pupil].min() # make it alsways positive
        #M[m, i, pupil] /= M[m, i, pupil].max()
        M[m, i] *= pupil #multiplying with the pupil function

plot_matrix(M[:,10,:,:], 7, 9, 'Actuator')

# %% fitting the influence functions with zernike functions
factor = 630e-6/2*np.pi # factor to convert rad to m
zern  = np.zeros((num_act, num_steps, size_mesh*size_mesh)) # contains the flatten zernike matrix
phase_retrieved = np.zeros((num_act, num_steps, size_mesh*size_mesh)) # contains the flatten phase matrix
H = np.zeros((num_steps, num_act, num_act)) # the individual influence matrix
for j in range(num_steps):
    for i in range(num_act):
        phase_retrieved[i, j, :] = M[i, j, :,:].flatten()
        zern[i, j, :] = ao.zernike_index(i+1, size_mesh, index='Noll', norm = 'Noll').flatten() * factor
    gamma = phase_retrieved[:, j, :]
    Z = zern[:, j, :]
    H[j] = Z@gamma.T
    
#H_avg = np.mean(H, axis=0)
H_median = np.median(H, axis=0)/150 # normalized median influence matrix in m/V
H_inv = np.linalg.inv(H_median) # inverse of the median influence matrix

# %% example of control matrix and zernikes and voltages
# voltage to zernike coefficients
pseudo_inverse = np.linalg.pinv(H_median) # 

us = np.zeros((63, 2*radius, 2*radius))
zerns = np.zeros((63, num_act))
for i in range(63):
    u = np.zeros(num_act)
    u[i] = 10 # V
    z = H_median@u # zernike coefficients
    zerns[i,:] = z
    us[i,:,:] = ao.zernike_multi(np.arange(2, 63), z, size_mesh, index='Noll', norm = 'Noll')

plot_matrix(us, 9, 7, 'influence functions')
plot_bar(zerns, 3, 3, 'actuator', ylabel='V', xlabel='Zernike modes')

#%%
# zernike coefficients to voltages
zerns = np.zeros((63,2*radius, 2*radius))
us = np.zeros((63, num_act))
for i in range(63):
    z = np.zeros(num_act)
    z[i] = 1 # m
    u = pseudo_inverse@z # V
    us[i] = u
    zerns[i, :, :] = ao.zernike_multi(np.arange(1, 63), H_median@u, 2*radius, index='Noll')
#ao.bar(np.arange(63), u)
#ao.show(ao.zernike_multi(np.arange(1, 63),H_avg@u, 64))

plot_matrix(zerns, 7, 9, 'Zernike modes')
plot_bar(us, 3, 3, 'Voltages')

# %%
# plot many subplots of a matrix
def plot_matrix(matrix, num_rows, num_cols, title, xlabel='', ylabel=''):
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 20))
    fig.suptitle(title)
    for i in range(num_rows):
        for j in range(num_cols):
            axs[i, j].imshow(matrix[i*num_cols+j], cmap='jet')
            axs[i, j].set_title(title +str(i*num_cols+j+1))
            axs[i, j].axis('off')
    plt.show()
#plot many bar subplots of a vector
def plot_bar(matrix, num_rows, num_cols, title, ylabel='', xlabel=''):
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 20))
    fig.suptitle(title)
    for i in range(num_rows):
        for j in range(num_cols):
            axs[i, j].bar(np.arange(len(matrix[i*num_cols+j])), matrix[i*num_cols+j])
            axs[i, j].set_title(title+str(i*num_cols+j+1))
            axs[i, j].set_ylabel(ylabel)
            axs[i, j].set_xlabel(xlabel)
    plt.show()
# %% Alex method -> eigenmodes and then 
