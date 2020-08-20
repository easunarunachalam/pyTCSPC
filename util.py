import cv2

import glob

# from lmfit import minimize, conf_interval, Parameters, Minimizer, fit_report
import lmfit

import matplotlib.pyplot as plt

import numpy as np

import os
from pathlib import Path, PurePath, PureWindowsPath

from scipy.interpolate import interp1d
from scipy.signal import fftconvolve, convolve

import sdtfile

import seaborn as sns

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

        # size of time bin (in nanoseconds)
        self.dt       = (self.times[1] - self.times[0])*1e9
        self.times    += self.times[1] - self.times[0]

        # number of frames in exposure (should be an integer)
        self.numscans = self.file.measure_info[0].MeasHISTInfo.fida_points[0]


        if flipud: self.data = np.flipud(self.data)
        if fliplr: self.data = np.fliplr(self.data)

        # (optional) store a mask indicating which pixels correspond to cells
        self.iscell   = None

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
                cv2.imwrite(self.filename_int_raw_str, rawsum_img)

            if corrsum:
                im = np.divide(im, self.numscans)
                if isinstance(illprof, np.ndarray) or isinstance(illprof, float) or isinstance(illprof, int):
                    im = np.divide(im, illprof)
                corrsum_img = im
                cv2.imwrite(self.filename_int_corr_str, corrsum_img*intensity_scale)

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
        elif isinstance(mask, np.ndarray):
            # add code to make sure dimensions are correct
            selected_decays = self.data[mask,:]
        elif mask == "cells":
            # check to make sure that segmentation is loaded
            if self.iscell is None:
                raise ValueError("Need to load segmentation")

            selected_decays = self.data[self.iscell,:]
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
            sns.lineplot(self.time(units="ns"), y=dc)
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
        return_objects=False
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

    sdt_filenames = [fn for fn in glob.glob(os.path.join(exp_folder,"**/*.sdt"), recursive=True) if ((exclude_folder not in fn) and (exclude_name not in fn))]

    timeseries    = [None]*len(sdt_filenames)
    sdt_list      = [None]*len(sdt_filenames)

    t0 = time.time()
    for i, fn in enumerate(sdt_filenames):

        sdt = SDT(fn)
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

        source_folder, filename = fn.rsplit(os.path.sep, maxsplit=1)

        if return_objects:
            sdt_list[i] = sdt

        t = time.time() - t0
        print("file {:d}/{:d} | elapsed time = {:3.2f} / {:3.2f} (est.) | last file: {:s} ".format(
                i+1, len(sdt_filenames), t, t*len(sdt_filenames)/(i+1), fn
            ), end="\r")

    if return_objects:
        return sdt_filenames, timeseries, sdt_list
    else:
        return sdt_filenames, timeseries

class decay_group:

    def __init__(self, data, irf, mask=None, start_bin=11, end_bin=237, modeltype="twoexp"):

        self.start_bin    = start_bin
        self.end_bin      = end_bin

        self.data         = data
        self.t_data       = data.time(units="ns")
        self.dc_data      = data.decay_curve(plot=False, normalize=False, mask=mask)
        self.nbins_data   = len(self.t_data)

        self.irf          = irf
        self.t_irf        = irf.time(units="ns")
        self.dc_irf       = irf.decay_curve(plot=False, normalize=True, trunc=True, bgsub=True)
        self.nbins_irf    = len(self.t_irf)

        self.modeltype    = modeltype

        self.params       = lmfit.Parameters()

        if self.modeltype is "oneexp":
            self.params.add('tau1'  , value=4.356242895126343  , min=0.001,  max=10.0)
            self.params.add('A'     , value=0.90985567510128021, min=0.90, max=1.0)
            self.params.add('shift' , value=9                  , min=-100, max=+100)
        elif self.modeltype is "twoexp":
            self.params.add('tau1'  , value=2.53  , min=0.001,  max=10.0)
            self.params.add('tau2'  , value=0.24  , min=0.001,  max=10.0)
            self.params.add('f'     , value=0.27  , min=0,      max=1)
            self.params.add('A'     , value=0.98  , min=0.90,   max=1.0)
            self.params.add('shift' , value=12     , min=-100,   max=+100)


    def oneexp(self, t, params):

        tau1  = params['tau1']
        A     = params['A']
        shift = params['shift']

        return (1-A) + A*np.exp(-t/tau1)

    def twoexp(self, t, params):

        tau1  = params['tau1']
        tau2  = params['tau2']
        f     = params['f']
        A     = params['A']
        shift = params['shift']

        return (1-A) + A*(f*np.exp(-t/tau1) + (1-f)*np.exp(-t/tau2))

    def model_conv_irf(self, t, params):

        # number of time bins (IRF time bin size) in a single laser period
        nbins_laser_period = int(self.irf.laser_period()/self.irf.dt)

        # time = one laser period
        times     = np.arange(0, self.irf.laser_period(), self.irf.dt)

        # values of model over this period
        if self.modeltype is "oneexp":
            model     = self.oneexp(times, params)
        elif self.modeltype is "twoexp":
            model     = self.twoexp(times, params)

        # duplicate model (add second period)
        dup_model = np.concatenate((model, model))

        # convolve model with IRF
        roll_irf  = np.roll(self.dc_irf, int(np.round(params["shift"])))
        cnv       = fftconvolve(roll_irf, dup_model, mode="full")*self.irf.dt
        # cnv       = fftconvolve(self.dc_irf, dup_model, mode="full")*self.irf.dt

        # keep the second period of the convolution
        p2        = cnv[nbins_laser_period+1:nbins_laser_period+1+self.nbins_irf]

        # collapse down to the same number of bins as the data
        rshp      = np.sum(np.reshape(p2, (self.nbins_data, len(p2)//self.nbins_data)), axis=1)

        # scale by total number of photons (in data) within time range of interest
        counts    = np.sum(self.dc_data[self.start_bin:(self.end_bin+1)])
        scaled    = counts*rshp/np.sum(rshp[self.start_bin:(self.end_bin+1)])

        # indices of requested times (first argument)
        t_idx     = np.round(np.divide(t, self.data.dt)).astype(int) - 1

        # print(scaled)
        return scaled[t_idx]

    def residual(self, params):
        return np.multiply(self.model_conv_irf(self.use_t, params) - self.use_data, self.fit_weight)

    def fit(
            self,
            method="leastsq",
            plot=False,
            datacolor="cornflowerblue",
            dataalpha=0.5,
            fitcolor="crimson"
        ):

        self.use_t, self.use_data = self.data.time()[self.start_bin:self.end_bin], self.dc_data[self.start_bin:self.end_bin]
        self.fit_weight = np.divide(1., np.sqrt(self.use_data))
        self.fit_weight[np.isinf(self.fit_weight)] = 0
        # self.fit_weight[np.isnan(self.fit_weight)] = 0


        # print(self.use_data)
        # print(self.fit_weight)
        m = lmfit.Minimizer(self.residual, self.params) #, args=(self.use_t, self.use_data, self.fit_weight))

        if method is "leastsq":
            out = m.minimize(ftol=1e-12, xtol=1e-12, method="leastsq")
        elif method is "emcee":
            out = m.minimize(method="emcee", burn=2000, steps=3000)
        elif method is "nelder":
            out = m.minimize(method="nelder")
        else:
            out = m.minimize(method=method)
            # raise SyntaxError("Unrecognized method. Available methods are `leastsq` and `emcee`.")

        self.final_vals = out.params.valuesdict()

        for p in out.params:
            out.params[p].stderr = abs(out.params[p].value * 0.1)

        ci = lmfit.conf_interval(m, out, sigmas=[1, 2])
        self.meanandCI = [(ci[str(key)][2][1], ci[str(key)][0][1], ci[str(key)][-1][1]) for key in ci.keys()]

        self.final_yhat = self.model_conv_irf(self.use_t, self.final_vals)
        scaled_residual = np.divide(self.use_data - self.final_yhat, np.sqrt(self.use_data))
        sr_hist, sr_bes = np.histogram(scaled_residual, bins=np.arange(-10,10,0.5))
        sr_bcs = sr_bes[1:] - 0.5*(sr_bes[1]-sr_bes[0])

        if plot:
            fig = plt.figure(constrained_layout=True, figsize=(18,6))
            w, h = [1, 1], [0.7, 0.3]
            spec = fig.add_gridspec(nrows=2, ncols=2, width_ratios=w, height_ratios=h)

            ax = fig.add_subplot(spec[0,0])
            ax.plot(self.t_data, self.dc_data, "o", color=datacolor, alpha=dataalpha, label="data")
            ax.plot(self.use_t, self.final_yhat, color=fitcolor, linestyle="-", label="fit")
            ax.legend()
            ax.set_xlabel(r"$t$/ns")
            ax.set_ylabel(r"counts")
            ax.set_yscale("log")

            ax = fig.add_subplot(spec[0,1])
            ax.plot(self.t_data, self.dc_data, "o", color=datacolor, alpha=dataalpha, label="data")
            ax.plot(self.use_t, self.final_yhat, color=fitcolor, linestyle="-", label="fit")
            ax.set_xlabel(r"$t$/ns")

            ax = fig.add_subplot(spec[1,0])
            ax.plot(self.use_t, scaled_residual, color=datacolor, alpha=dataalpha*2)
            ax.axhline(y=0, color="gray")
            # ax.set_ylim([-0.05,0.05])
            ax.set_xlabel(r"$t$/ns")
            ax.set_ylabel(r"scaled residual")

            ax = fig.add_subplot(spec[1,1])
            ax.bar(sr_bcs, sr_hist, width=sr_bcs[1]-sr_bcs[0], color=datacolor, alpha=dataalpha)
            ax.set_xlabel(r"scaled residual")
            ax.set_ylabel(r"counts")
            plt.show()

        return self.meanandCI, (self.use_t, self.final_yhat)
