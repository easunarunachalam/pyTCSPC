from datetime import datetime
import glob
import imageio as iio
import lmfit
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path, PurePath, PureWindowsPath
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve, convolve
import sdtfile
import sys
import time

sys.path.append("../")
import util


def sdt_to_images(
        exp_folder,
        exclude_folder="!!",
        exclude_name="!!",
        im_save=True,
        im_format="tiff",
        im_folder="",
        illprof=1,
        intensity_scale=100,
        tau_scale=10,
        return_images=False,
        return_objects=False,
        return_all_fns=False,
        only_first_n=None,
        regenerate=False
    ):
    '''
    generate intensity images from all sdt files

    Parameters
    ----------
    exp_folder: folder containing all images to be converted (e.g. a day's experiment folder)
    exclude_folder (str, default="illprof"): exclude files which have this string in their path, e.g. for illumination profile SDTs
    im_format (str; default="tiff"): image format
    im_folder (str; default="Pos0"): image folder, name of new folder containing all images, the same name for all samples. One folder for all images in one sample
    illprof (float/np.ndarray; default=1): illumination profile, default is uniform. If non-scalar, should have same dimensions as images to be converted.
    intensity_scale (float; default = 1): going to divide all pixel intensities by the number of scans; we can then multiply all pixel intensities by some factor intensity_scale to use more of the range of 8-bit images
    flipud (bool): reverse image from up to down
    fliplr (bool): reverse image from left to right

    Returns
    -------
    sdt_filenames: list of names of sdt files that have been converted to intensity images

    '''

    # get list of sdt files to process

    sdt_filenames_all = sorted([fn for fn in Path(exp_folder).glob("**/*.sdt") if fn.is_file()])

    if not regenerate:
        sdt_filenames_use = []

        for sdt_fn in sdt_filenames_all:
            img_fn = Path.joinpath(sdt_fn.parent, sdt_fn.stem + "_intensity_corr.tiff")
            if not img_fn.is_file():
                sdt_filenames_use.append(sdt_fn)
    else:
        sdt_filenames_use = sdt_filenames_all



    # image generation

    timeseries    = [None]*len(sdt_filenames_use)
    sdt_list      = [None]*len(sdt_filenames_use)

    clearline = 120*" "

    if len(sdt_filenames_use) > 0:
        t0 = time.time()
        for i, sdt_fn in enumerate(sdt_filenames_use):

            t = time.time() - t0
            if i == 0:
                est_time = 0
            else:
                est_time = t*len(sdt_filenames_use)/(i)
            print("file {:d}/{:d} | elapsed time = {:3.2f} / {:3.2f} (est.) | current file: {:s} ".format(
                    i+1, len(sdt_filenames_use), t, est_time, str(sdt_fn)
                ) + clearline, end="\r")

            sdt = util.SDT(sdt_fn)
            im = sdt.image(
                        rawsum=im_save,
                        corrsum=im_save,
                        meantau=False,
                        adjust_intensity=True,
                        illprof=illprof,
                        intensity_scale=intensity_scale,
                        tau_scale=tau_scale
                    )

            timeseries[i] = im

            if return_objects:
                sdt_list[i] = sdt

            if only_first_n and (i+1)>=only_first_n:
                break

        t = time.time() - t0
        print(clearline + "\r{:d} files converted | elapsed time = {:3.2f}".format(i+1, t) + clearline, end="\n")

    if return_images and return_objects:
        return sdt_filenames_use, timeseries, sdt_list
    elif return_images:
        return sdt_filenames_use, timeseries
    elif return_objects:
        return sdt_filenames_use, sdt_list
    elif return_all_fns:
        return sdt_filenames_all
    else:
        return sdt_filenames_use
