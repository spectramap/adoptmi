from AO import*

import numpy as np
import matplotlib.pyplot as plt
import aotools as ao

def create_new_control_matrix(Zernikes, phi_new):
    """calculates the new control matrix"""
    act_no = phi_new.shape[0]
    B_new = np.zeros((Zernikes.shape[0], act_no))
    pupil = create_pupil(Zernikes.shape[1], Zernikes.shape[1])
    area = np.sum(pupil)
    for m in range(Zernikes.shape[0]):
        for n in range(act_no):
             Z = Zernikes[m,:,:] * pupil
             B_new[m,n] = np.sum(Z * phi_new[n,:,:] * pupil)/area
    return B_new

def generate_zernikes(N, noll_maxidx):
    """generates Zernike modes with N pixels side length"""
    zernikes = np.zeros((noll_maxidx, N, N))
    a = np.zeros(noll_maxidx)
    
    for m in range(noll_maxidx):
        a = np.zeros(noll_maxidx)
        a[m] = 1 #setting the amplitude of the Zernike mode to 1
        zernikes[m,:,:] = zernike_index(m, N, index = 'OSA')
    return zernikes

def generate_phase_from_control_matrix(B, sort_idx):
    N = 100
    act_no = B.shape[1]
    phi = np.zeros((act_no, N, N))
   
    for m in range(act_no):
        v = np.zeros(B.shape[1])
        v[m] = 1
        a_z = B @ v #if B has been created with OSA-Zernikes, a_z will be a vector of OSA-Zernike coefficients
        phi[m,:,:] = zernike_multi(sort_idx, a_z[sort_idx], N, index = 'Noll')
        #print(zernike_multi(sort_idx, [1, 1], N, index = 'OSA'))
        #phi[m,:,:] = ao.functions.zernike.phaseFromZernikes(a_z[sort_idx], N, norm='rms')
    return phi


plt.ion()
path = "C:\\Users\\q125jm\\Documents\\Python Scripts\\Influence_matrix.npy"
full_inf_matrix = np.load(path)

inf_matrix = full_inf_matrix[:, 3, :,:]

##
dia_DPP = 10e-3 #pupil diameter of the DPP
dia_raman = 9e-3 #pupil diameter of the Raman microscope

phi_new = np.zeros((63, 90, 90))
for i in range(63):
    phi_new[i,:,:] = crop_phases_to_smaller_pupil(dia_DPP, dia_raman, inf_matrix[i,:,:])

Zernikes = generate_zernikes(phi_new.shape[1], 93) #calculating Zernikes for the new pupil diameter
B_new = create_new_control_matrix(Zernikes, phi_new)

path_trial = "C:\\Users\\q125jm\\Documents\\Raman\\InfMat.mat"

#read matlab influence matrix
def read_matlab_influence_matrix(path):
    import scipy.io as sio
    mat_contents = sio.loadmat(path)
    inf_matrix = mat_contents['Influence_mat']
    return inf_matrix

inf_matrix_pou = read_matlab_influence_matrix(path_trial)

phi_single = generate_phase_from_control_matrix(inf_matrix_alex, np.arange(1,64))

plt.figure(), plt.imshow(phi_single[7,:,:])

###

plt.ion()
reference = ao.zernike_noll(4, 64)