import imageio
import matplotlib.pyplot as plt
import numpy as np
import sdtfile as bhsdt
from skimage.filters import rank, gaussian
from skimage.morphology import watershed, disk

from colorbar import colorbar


def smooth_illprof(intensity_illprof, plot=False, saveplot=False):
    intensity_illprof_scaled = np.uint8((2**8)**np.divide(intensity_illprof, np.max(intensity_illprof)))
    blur_median = rank.median(intensity_illprof_scaled, selem=disk(6))*1e4
    blur_gaussian = gaussian(blur_median, sigma=15)*1e4

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
