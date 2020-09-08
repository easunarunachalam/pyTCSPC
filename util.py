from datetime import datetime
import glob
import imageio as iio
import lmfit
import matplotlib.pyplot as plt
import numpy as np
# import cv2
import os
from pathlib import Path, PurePath, PureWindowsPath
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve, convolve
import sdtfile
import sys
import time



class SDT(object):
    '''
    .sdt file object

    Parameters (constructor)
    ------------------------
    filename (str): name of .sdt file to read in
    block (int; default=0): index of channel to read. To use another channel, create another instance.
    '''
    def __init__(
        self,
        filename,
        block=0,
        flipud=True,
        fliplr=False
    ):
        self.filename              = PurePath(filename)
        self.parent_dir            = self.filename.parents[0]

        self.filename_int_raw      = PurePath.joinpath(self.parent_dir, self.filename.stem + "_intensity_raw.tiff")
        self.filename_int_corr     = PurePath.joinpath(self.parent_dir, self.filename.stem + "_intensity_corr.tiff")
        self.filename_tau          = PurePath.joinpath(self.parent_dir, self.filename.stem + "_lifetime.tiff")
        self.filename_str          = str(self.filename)
        self.filename_int_raw_str  = str(self.filename_int_raw)
        self.filename_int_corr_str = str(self.filename_int_corr)
        self.filename_tau_str      = str(self.filename_tau)

        self.file                  = sdtfile.SdtFile(self.filename_str)
        self.data                  = self.file.data[block]
        self.times                 = self.file.times[block]
        self.get_acqtime()

        # size of time bin (in nanoseconds)
        self.dt       = (self.times[1] - self.times[0])*1e9
        self.times    += self.times[1] - self.times[0]

        # number of frames in exposure (should be an integer)
        self.numscans = self.file.measure_info[0].MeasHISTInfo.fida_points[0]


        if flipud: self.data = np.flipud(self.data)
        if fliplr: self.data = np.fliplr(self.data)

        # (optional) store a mask indicating which pixels correspond to cells
        self.iscell   = None

    def get_acqtime(self):
        '''
        get acquisition time (wall clock time) for SDT
        '''
        info_lines   = self.file.info.split("\n")
        hhmmss_str   = [i.split(":") for i in info_lines if i.startswith('  Time')][0][1:]
        hhmmss       = [int(j) for j in hhmmss_str]
        yyyymmdd_str = [i.split(":") for i in info_lines if i.startswith('  Date')][0][1]
        yyyymmdd     = [int(i) for i in yyyymmdd_str.split("-")]
        self.acqtime = datetime(yyyymmdd[0], yyyymmdd[1], yyyymmdd[2], hhmmss[0], hhmmss[1], hhmmss[2])

    def __add__(self, other):
        self.data += other.data

        return self

    def image(
            self,
            rawsum=True,
            corrsum=True,
            meantau=False,
            adjust_intensity=False,
            illprof=None,
            intensity_scale=100,
            tau_scale=1
        ):
        '''
        create intensity image by summing photon counts over time for each pixel

        Parameters
        ----------
        rawsum : bool
            if True, produce raw intensity image (not correcting for number of frames)
        corrsum : bool
            if True, produce corrected intensity image (accounting for number of frames, and the illumination profile if provided)
        meantau : bool
            if True, produce mean lifetime image
        adjust_intensity : bool
            adjust intensities according to number of scans and specified intensity scale
        illprof : float or 2-d numpy.ndarray with same x, y dimensions self.data
            illumination profile
        '''

        rawsum_img, corrsum_img, meantau_img = None, None, None

        if rawsum or corrsum:
            im = self.data.sum(axis=2)

            if rawsum:
                rawsum_img = im.astype("int")
                iio.imwrite(self.filename_int_raw_str, rawsum_img)

            if corrsum:
                im = np.divide(im, self.numscans)
                if isinstance(illprof, np.ndarray) or isinstance(illprof, float) or isinstance(illprof, int):
                    im = np.divide(im, illprof)
                corrsum_img = (im*intensity_scale).astype("int")
                iio.imwrite(self.filename_int_corr_str, corrsum_img)

        if meantau:

            def avg_time(a):
                a_tot = np.sum(a)
                if a_tot == 0:
                    return np.nan
                else:
                    a_norm = np.divide(a, a_tot)
                    a_x_time = np.multiply(a_norm, self.times)
                    return np.sum(a_x_time)

            meantau_img = np.apply_along_axis(avg_time, 2, self.data)
            meantau_img[np.isnan(meantau_img)] = 0

            cv2.imwrite(self.filename_tau_str, 1e9*meantau_img*tau_scale)

        return rawsum_img, corrsum_img, meantau_img

    def hist(self, mode="sum", minval=0, maxval=200, interval=1):
        '''
        calculate histogram of pixel values

        Parameters
        ----------
        mode (str; default="sum"): option passed to image function
        maxval (int or float): maximum value in histogram
        minval (int or float): minimum value in histogram
        intval (int or float): histogram bin width
        '''

        im = self.image(mode=mode)
        flat_val = np.reshape(im, (np.size(im),1))
        counts, bin_edges = np.histogram(flat_val, bins=np.arange(minval,maxval,interval)-0.5)
        bin_centers = bin_edges[:-1] + 0.5

        return counts, bin_centers

    def total_photon_count(self):
        return self.data.sum()

    def time(self, units="ns"):
        '''
        return time values for bins in decay curve
        Parameters
        ----------
        units (str, default="ns"): if "ns", use units of nanoseconds; if not specified, use default units (seconds)
        '''

        if units == "ns":
            return self.times*1e9
        else:
            return self.times

    def decay_curve(self, mask=None, normalize=False, plot=False, trunc=False, bgsub=False):
        '''
        return histogram of photon lifetimes

        Parameters
        ----------
        mask (variable; default=None):
            if None, use decay curves for all pixels
            if "cells", use the stored mask self.iscell (must have already calculated/imported the mask)
            if a binary nxmx1 np.ndarray, where n,m are the x,y dimensions of the image, use the decay curves for elements that are True
        nbins (int; default=256): number of time bins to use for decay curve. Original number of bins should be a mutiple of this number.
        normalize (bool; default=False): if True, divide all counts by the total number of counts in all bins
        plot (bool; default=False): if True, plot the decay curve

        '''

        if mask is None:
            selected_decays = self.data
            dims = self.data.shape
            self.npx = dims[0]*dims[1]
        elif isinstance(mask, np.ndarray):
            # add code to make sure dimensions are correct
            selected_decays = self.data[mask,:]
            self.npx = np.sum(mask)
        elif mask == "cells":
            # check to make sure that segmentation is loaded
            if self.iscell is None:
                raise ValueError("Need to load segmentation")

            selected_decays = self.data[self.iscell,:]
            self.npx = np.sum(self.iscell)
        else:
            raise SyntaxError("Invalid mask type")


        # combine decays for all selected pixels
        # (sum values from all decays in each time bin)
        n_time_bins = selected_decays.shape[-1]
        dc = selected_decays.reshape(-1, n_time_bins).sum(axis=0).astype(float)

        if normalize:
            dc = np.divide(dc, np.sum(dc) *self.dt)

        if bgsub:
            is_background = (self.time() > 7) & (self.time() < 9)
            background_time = self.time()[is_background]
            background = dc[is_background]
            mean_background = np.mean(background)
            dc -= mean_background
            dc[dc < 0] = 0

        if trunc:
            is_peak_region = (self.time() >= 1.5) & (self.time() <= 3)
            dc[np.logical_not(is_peak_region)] = 0

        if plot:
            plt.plot(self.time(units="ns"), dc)
            plt.xlabel(r"$t$/ns")
            if normalize:
                plt.ylabel(r"frequency")
            else:
                plt.ylabel(r"photons")
            plt.yscale("log")

        return dc

    def laser_period(self):
        '''
        return laser period (in nanoseconds)
        '''
        mi = self.file.measure_info[0]
        return np.mean((1/mi["StopInfo"]["min_sync_rate"][0], 1/mi["StopInfo"]["max_sync_rate"][0]))*1e9

    def load_ilastik_sgm(self, fname_prob_map=None, channel_cell=0, p_threshold_cell=0.5):

        if fname_prob_map is None:
            fname_prob_map = os.path.splitext(self.filename)[0] + "_Probabilities.npy"

        segmentation      = np.load(fname_prob_map)
        self.iscell       = segmentation[:,:,channel_cell] > p_threshold_cell

    def cell_px_intensities(self, return_all_values=False, print_result=False):

        if self.iscell is None:
            raise TypeError("Need to load segmentation.")
            # sys.exit()

        cell_px = self.image()[self.iscell]

        if print_result:
            print("Total photons from cells =", np.sum(cell_px))
            print("Mean intensity of cell pixels =", np.mean(cell_px))

        if return_all_values:
            return np.mean(cell_px), np.sum(cell_px), cell_px
        else:
            return np.mean(cell_px), np.sum(cell_px)


from mpl_toolkits.axes_grid1 import make_axes_locatable

# for creating appropriately-sized colorbars for subplots
def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)
