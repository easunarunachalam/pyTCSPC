import imageio
import matplotlib.pyplot as plt
import numpy as np
import sdtfile as bhsdt

from skimage.filters import rank, gaussian
from skimage.morphology import watershed, disk
from skimage.transform import rescale

from util import colorbar


def smooth_illprof(intensity_illprof, rescale_factor=None, sigma=15, diskr=6, plot=False, saveplot=False):

    if rescale_factor:
        intensity_illprof_rescaled = rescale(intensity_illprof, rescale_factor, anti_aliasing=False)

    intensity_illprof_scaled = np.uint8((2**8)**np.divide(intensity_illprof_rescaled, np.max(intensity_illprof_rescaled)))

    blur_median = rank.median(intensity_illprof_scaled, selem=disk(diskr))*1e4
    blur_gaussian = gaussian(blur_median, sigma=sigma)*1e4

    blur = blur_gaussian
    blur = np.divide(blur, np.max(blur))

    if plot:

        fig, ax = plt.subplots(figsize=(15,6), nrows=1, ncols=2)

        p0 = ax[0].imshow(intensity_illprof_scaled)
        ax[0].set_title("Original")
        colorbar(p0)

        p1 = ax[1].imshow(blur, vmin=0, vmax=1)
        ax[1].set_title("Smoothed")
        colorbar(p1)

    return blur
