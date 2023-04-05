import os
import sys
import bartview
import numpy as np
from volume_processing import display2d, imshow4

import matplotlib.pyplot as plt
from bart import bart
from cfl import *

from bartpy import bartpy as bp

# sqrt sum-of-squares of k-space
und2x2 = readcfl('data/bart/und2x2')
ksp_rss = bp.rss(und2x2, 8)

# zero-filled reconstruction sqrt-sum-of-squares
zf_coils = bp.fft(und2x2, 6, i=True)
zf_rss = bp.rss(zf_coils, 8)

ksp_rss = np.squeeze(ksp_rss)
zf_coils = np.squeeze(zf_coils)
zf_rss = np.squeeze(zf_rss)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(abs(ksp_rss)**0.125, cmap='gray')
ax1.set_title('k-space')
ax2.imshow(abs(zf_rss), cmap='gray')
ax2.set_title('zero-filled recon')
plt.show(block=True)

calmat = bp.calmat(und2x2,k=6,r=20)
(U, SV, VH) = bp.svd(calmat)

plt.plot(SV)
plt.title('Singular Values of the Calibration Matrix')
plt.show(block=True)

(calib, emaps) = bp.ecalib(und2x2, ev_maps=True, r=20)
sens = bp.slice(calib, 4,0)
sens_maps = np.squeeze(sens)

imshow4(np.abs(sens_maps), N_YX=[2,4], block=True, title='Magnitude ESPIRiT 1st Set of Maps')
imshow4(np.angle(sens_maps), N_YX=[2,4], block=True, title='Phase ESPIRiT 1st Set of Maps')

full = readcfl('data/bart/full')
coilimgs = bp.fft(full, 6, i=True)

coil_imgs = np.squeeze(coilimgs)
imshow4(np.abs(coil_imgs), N_YX=[2,4], block=True, title='Coil images')

emaps = np.squeeze(emaps)
imshow4(emaps,N_YX=[1,2], block=True, title='First Two Eigenvalue Maps')

# SENSE reconstruction using ESPIRiT maps (using the generalized parallel imaging compressed sensing tool)
#reco = bart(1, 'pics', und2x2, sens);
reco = bp.pics(und2x2, sens)

sense_recon = np.squeeze(reco)
imshow4(abs(sense_recon), block=True, title='ESPIRiT Reconstruction')

# Evaluation of the coil sensitivities.
#
# Computing error from projecting fully sampled error
# onto the sensitivities. This can be done with one
# iteration of POCSENSE.
proj = bp.pocsense(full, sens, r=0.0, i=1)

# Compute error and transform it into image domain and combine into a single map.
errimgs = bp.fft((full - proj), 6, i=True)
errsos_espirit = bp.rss(errimgs, 8)

#
# For comparison: compute sensitivities directly from the center.
sens_direct = bp.caldir(und2x2, cal_size=20)

# Compute error map.
proj = bp.pocsense(full, sens_direct, r=0.0, i=1)
errimgs = bp.fft((full - proj), 6, i=True)
errsos_direct = bp.rss(errimgs, 8)


errsos_espirit = np.squeeze(errsos_espirit)
errsos_direct = np.squeeze(errsos_direct)

imshow4(abs(np.concatenate((errsos_direct, errsos_espirit), axis=1)),N_YX=[1,1], title='Projection Error (direct calibration vs ESPIRiT)');

#kspace = cfl.readcfl("./ksp_img_pc")
#sensitivities = bart(1, 'ecalib', kspace);
##zf_coils = bart(1, 'fft -i 3', kspace)
#bartview.plt.imshow(np.abs(zf_coils), cmap='gray')
#bartview.plt.show(block=True)



# Example 2: Reconstruction of undersampled data with small FOV.
# This example uses a undersampled data set with a small FOV. The image reconstructed using ESPIRiT is compared to an image reconstructed with SENSE. By using two sets of maps, ESPIRiT can avoid the central artifact which appears in the SENSE reconstruction.

# Zero padding to make square voxels since resolution in x-y for this
# data set is lower in phase-encode than readout
smallfov = readcfl('data/bart/smallfov')
smallfov = bp.resize(smallfov, True, 2, 252)
# Direct calibration of the sensitivities from k-space center for SENSE
sensemaps = bp.caldir(smallfov, 20)
# SENSE reconstruction
sensereco = bp.pics(smallfov, sensemaps, r=0.01)

# ESPIRiT calibration with 2 maps to mitigate with aliasing in the calibration
espiritmaps = bp.ecalib(smallfov, r=20, m=2)
# ESPIRiT reconstruction with 2 sets of maps
espiritreco = bp.pics(smallfov, espiritmaps, r=0.01)
# Combination of the two ESPIRiT images using root of sum of squares
espiritreco_rss = bp.rss(espiritreco, 16)

# ESPIRiT calibration with 1 map to mitigate with aliasing in the calibration
espiritmaps1 = bp.ecalib(smallfov, r=20, m=1)
# ESPIRiT reconstruction with 1 sets of maps
espiritreco1 = bp.pics(smallfov, espiritmaps1, r=0.01)
# Combination of the two ESPIRiT images using root of sum of squares
espiritreco1_rss = bp.rss(espiritreco1, 16)

espirit_maps = np.squeeze(espiritmaps)
imshow4(abs(espirit_maps), title='The two sets of ESPIRiT maps')

# SENSE image:
reco1 = np.squeeze(sensereco)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(abs(reco1), cmap='gray')
ax1.set_title('SENSE Reconstruction')

# ESPIRiT image 2 maps:
reco2 = np.squeeze(espiritreco_rss)
ax2.imshow(abs(reco2), cmap='gray')
ax2.set_title('ESPIRiT Reconstruction from 2 maps')

# ESPIRiT image 1 map:
reco3 = np.squeeze(espiritreco1_rss)
ax3.imshow(abs(reco3), cmap='gray')
ax3.set_title('ESPIRiT Reconstruction from 1 map')

plt.show(block=True)


# Example 3: Compressed Sensing and Parallel Imaging
# This example demonstrates L1-ESPIRiT reconstruction of a human knee. Data has been acquired with variable-density poisson-disc sampling.
# A visualization of k-space data

knee = readcfl('data/bart/knee')
ksp_rss = bp.rss(knee, 8)
ksp_rss = np.squeeze(ksp_rss)
imshow4(abs(ksp_rss)**0.125, title='k-space')

# Root-of-sum-of-squares image
knee_imgs = bp.fft(knee, 6, i=True)
knee_rss = bp.rss(knee_imgs, 8)

# ESPIRiT calibration (one map)
knee_maps = bp.ecalib(knee, c=0.0, m=1)

# l1-regularized reconstruction (wavelet basis)
knee_l1 = bp.pics(knee, knee_maps, l=1, r=0.01)

# Results
knee_rss = knee_rss / 1.5E9

image = np.concatenate((np.squeeze(knee_rss), np.squeeze(knee_l1)), axis=1)
imshow4(abs(image), title='Zero-filled and Compressed Sensing/Parallel Imaging')

# Example 4: Basic Tools
# Various tools are demonstrated by manipulating an image.
knee_l1 = readcfl('data/bart/knee_l1')

# Zero pad
knee2 = bp.resize(knee_l1, True, 1, 300, 2, 300)

# Switch dimensions 1 and 2
tmp = bp.transpose(knee2, 1, 2)

# Scale by a factor of 0.5
tmp2 = bp.scale(tmp, 0.5)

# Join original and the transposed and scaled version along dimension 2.
joined = bp.join(knee2, tmp2, 2)

# Flip 1st and 2nd dimension (2^1 + 2^2 = 6)
tmp = bp.flip(joined, 6)

# Join flipped and original along dimension 1.
big = bp.join(joined, tmp, 1)

# Extract sub-array
tmp = bp.extract(big, 1, 150, 449)
small = bp.extract(tmp, 2, 150, 449)

# Circular shift by 115 pixels
tmp = bp.circshift(small, 1, 150)
shift = bp.circshift(tmp, 2, 150)

# Show the final result.
imshow4(abs(np.squeeze(shift)))

# non-Cartesian MRI using BART

# Generate k-space trajectory with 64 radial spokes
traj_rad = bp.traj(x=512, y=64, r=True)

# 2x oversampling
traj_rad2 = bp.scale(traj_rad, 0.5)

# simulate eight-channel k-space data
ksp_sim = bp.phantom(k=True,s=8,t=traj_rad2)

# increase the reconstructed FOV a bit
traj_rad2 = bp.scale(traj_rad, 0.6)


# inverse gridding
igrid = bp.nufft(traj_rad2, ksp_sim, i=True, t=True)

# channel combination
reco1 = bp.rss(igrid, 8)

# reconstruct low-resolution image and transform back to k-space
lowres_img = bp.nufft(traj_rad2, ksp_sim, i=True, d="24:24:1", t=True)
lowres_ksp = bp.fft(lowres_img, 7, u=True)

# zeropad to full size
ksp_zerop = bp.resize(lowres_ksp, True, 0, 308, 1, 308)

# ESPIRiT calibration
sens = bp.ecalib(ksp_zerop, m=1)

# non-Cartesian parallel imging
reco2 = bp.pics(ksp_sim, sens, S=True, r=0.001,t=traj_rad2,d=4)

imshow4( abs(np.squeeze(np.concatenate((reco1, reco2), axis=1))), title ='Inverse Gridding vs Parallel Imaging')
