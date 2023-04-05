import math
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import skimage
import scipy.ndimage as ndimage

def ClipRescale(v, a_min_normalized=0.0, a_max_normalized=1.0):
    r = v.copy()
    v_max = np.amax(v)
    vessel_min = a_min_normalized * v_max
    vessel_max = a_max_normalized * v_max
    r[:,:,:] = np.clip(v, a_min=vessel_min, a_max = vessel_max)
    r[:,:,:] = (r - vessel_min) / (vessel_max - vessel_min)

    return r

def volume_padding(v):
    # padd first and last slice with zeros
    # 3D texture rendering may have problem in first and last slice during rendering
    # and make it brighter than expected
    vol_padded = np.zeros( (v.shape[0], v.shape[1], v.shape[2]+2), dtype=np.float32)
    vol_padded[:,:,1:vol_padded.shape[2]-1] = v
    
    return vol_padded

def display2d(v, N_XY=[], block=True, title=''):
    s = v.shape
    if (N_XY == []):
        if (len(v.shape)==3):
            NX = math.ceil(math.sqrt(s[2]));  NY = NX
            NS = s[2]
        elif(len(v.shape)==4):
            NX = v.shape[2]
            NY = v.shape[3]
            #NX = math.ceil(math.sqrt(s[2]));  NY = NX
            NS = s[2] * s[3]
            v = np.reshape(v, (v.shape[0], v.shape[1], NS))
        else:
            NY = NX = 1
            NS = 1
            v = np.reshape(v,(v.shape[0],v.shape[1],NS))
    else:
        NX = N_XY[0]
        NY = N_XY[1]
        NS = s[2] if (len(v.shape)>2) else 1

    canvas = np.zeros((v.shape[0]*NX, v.shape[1]*NY), dtype=np.float32)

    for j in range(NY):
        for i in range(NX):
            os = j * NX + i
            if (os < NS):
                canvas[ i*v.shape[0]:(i+1)*v.shape[0], j*v.shape[1]:(j+1)*v.shape[1]] = v[:,:,os]

    plt.imshow(canvas, cmap='gray')
    plt.title(title)
    plt.draw()
    plt.pause(0.001)
    plt.show(block=block)
    plt.show()

def imshow4(v, N_YX=[], block=True, title='', window_title='', min_border=True):
    if (N_YX==[]):
        if (len(v.shape)==2):
            N_YX = [1,1]
            NS = 1
        elif (len(v.shape)==3):
            #N_YX = [1,v.shape[2]]
            # Auto Square
            N = math.ceil(math.sqrt(v.shape[2]))
            N_YX = [N,N]
            N_YX[1] = N-1 if (N*(N-1)==v.shape[2]) else N
            NS = v.shape[2]
        elif (len(v.shape)==4):
            N_YX = [v.shape[3],v.shape[2]]
            NS = v.shape[3] * v.shape[2]
        else:
            # Cannot display image with 1 dimension
            return

        v = np.reshape(v, (v.shape[0], v.shape[1], NS))
    else:
        NS = N_YX[0] * N_YX[1]
        v = np.reshape(v, (v.shape[0], v.shape[1], NS))

    canvas = np.zeros((v.shape[0]*N_YX[0], v.shape[1]*N_YX[1]), dtype=np.complex64)

    for j in range(N_YX[1]):
        for i in range(N_YX[0]):
            os = j * N_YX[0] + i
            if (os < NS):
                canvas[ i*v.shape[0]:(i+1)*v.shape[0], j*v.shape[1]:(j+1)*v.shape[1]] = v[:,:,os]

    if (min_border):
        fig = plt.figure()
        fig.frameon = False
        ax = fig.add_axes((0, 0, 1, 1))
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    plt.box(False)
    plt.imshow(abs(canvas), cmap='gray')
    plt.title(title)
    plt.pause(0.01)
    plt.draw()
    fig = plt.gcf()
    fig.canvas.manager.set_window_title(window_title)
    plt.show(block=block)

def extract_brain_3d_mask(v):
    # blur the volume to denoise
    blurred_vol = skimage.filters.gaussian(v, sigma=3.0)
    # perform automatic thresholding
    t = skimage.filters.threshold_otsu(blurred_vol)
    print("Found automatic threshold t = {}.".format(t))

    # create a binary mask with the threshold found by Otsu's method
    threshold_factor = 1.0  # manual adjust
    binary_mask = blurred_vol > (t * threshold_factor)  # return as binary type

    return binary_mask

def extract_brain_3d_mask_2d_stack(v):
    binary_mask = np.zeros(v.shape, dtype=np.bool)

    # blur the volume to denoise
    blurred_vol = skimage.filters.median(v)

    s = v.shape[2]
    for i in range(s):
        # perform automatic thresholding
        t = skimage.filters.threshold_otsu(blurred_vol[:,:,i])
        #print("Found automatic threshold t = {}.".format(t))

        # create a binary mask with the threshold found by Otsu's method
        threshold_factor = 1.0  # manual adjust in tri-modal cases
        binary_mask[:,:,i] = blurred_vol[:,:,i] > (t * threshold_factor)  # return as binary type

    return binary_mask

def fill_holes_3d_as_2d_stack(vol_mask):
    r = np.zeros_like(vol_mask)
    s = vol_mask.shape[2]

    for i in range(s):
        # additional dilation/erosion pair to prevent some void are in brain
        # such as sinus, become and disconnected hole
        r[:,:,i] = ndimage.binary_dilation(vol_mask[:,:,i], iterations=8)
        r[:,:,i] = ndimage.binary_fill_holes(r[:,:,i])
        r[:,:,i] = ndimage.binary_erosion(r[:,:,i], iterations=8)

    return r

def trim_mask_3d_as_2d_stack(vol_mask, iterations=1):
    r = np.zeros_like(vol_mask)
    s = vol_mask.shape[2]

    for i in range(s):
        r[:,:,i] = ndimage.binary_erosion(vol_mask[:,:,i], iterations=iterations)

    return r