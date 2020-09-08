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


class decay_group:

    def __init__(self, data, irf, mask=None, start_bin=11, end_bin=237, modeltype="twoexp"):

        self.start_bin    = start_bin
        self.end_bin      = end_bin

        self.data         = data
        self.t_data       = data.time(units="ns")
        self.dc_data      = data.decay_curve(plot=False, normalize=False, mask=mask)
        self.nbins_data   = len(self.t_data)
        self.nphot_data   = np.sum(self.dc_data)
        self.intensity    = self.nphot_data / self.data.npx

        self.irf          = irf
        self.t_irf        = irf.time(units="ns")
        self.dc_irf       = irf.decay_curve(plot=False, normalize=True, trunc=True, bgsub=True)
        self.nbins_irf    = len(self.t_irf)
        self.nphot_irf    = np.sum(self.dc_irf)

        self.modeltype    = modeltype
        self.params       = lmfit.Parameters()

        if self.modeltype is "oneexp":
            self.params.add('tau1'  , value=4.356242895126343  , min=0.000001,  max=100.0)
            self.params.add('A'     , value=0.90985567510128021, min=0.90, max=1.0)
            self.params.add('shift' , value=9                  , min=-100, max=+100)
        elif self.modeltype is "twoexp":
            self.params.add('tau1'  , value=2.53  , min=0.000001,  max=100.0)
            self.params.add('tau2'  , value=0.24  , min=0.000001,  max=100.0)
            self.params.add('f'     , value=0.27  , min=0,      max=1)
            self.params.add('A'     , value=0.98  , min=0.90,   max=1.0)
            self.params.add('shift' , value=12     , min=-100,   max=+100)
        else:
            raise ValueError("Invalid model type.")

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

        m = lmfit.Minimizer(self.residual, self.params) #, args=(self.use_t, self.use_data, self.fit_weight))

        if method is "leastsq":
            out = m.minimize(ftol=1e-12, xtol=1e-12, method="leastsq")

        elif method is "emcee":

            out = m.minimize(method="dual_annealing")
            self.final_vals = out.params.valuesdict()
            for key in self.params.keys():
                self.params[str(key)].value = self.final_vals[str(key)]

            out = m.minimize(method="emcee", burn=500, steps=1000)
            # out = m.minimize(method="emcee", burn=2000, steps=3000)

        else:
            out = m.minimize(method=method)
            # raise SyntaxError("Unrecognized method. Available methods are `leastsq` and `emcee`.")

        self.final_vals = out.params.valuesdict()

        # need to initialize std err value
        for p in out.params:
            out.params[p].stderr = abs(out.params[p].value * 1e-1)

        self.ci = lmfit.conf_interval(m, out, sigmas=[1, 2])
        self.meanandCI = np.array([[self.ci[str(key)][2][1], self.ci[str(key)][0][1], self.ci[str(key)][-1][1]] for key in self.ci.keys()])

        if self.modeltype is "twoexp":

            # [print(key) for key in self.ci.keys()]
            tau1, tau2, f, A, shift = self.meanandCI[0], self.meanandCI[1], self.meanandCI[2], self.meanandCI[3], self.meanandCI[4]

            if tau1[0] > tau2[0]:
                self.meanandCI = np.array([tau1, tau2, f, A, shift])
            if tau1[0] < tau2[0]:
                self.meanandCI = np.array([tau2, tau1, 1-f, A, shift])
                self.meanandCI[2] = np.array([self.meanandCI[2][0], self.meanandCI[2][2], self.meanandCI[2][1]])

        self.final_yhat = self.model_conv_irf(self.use_t, self.final_vals)
        scaled_residual = np.divide(self.use_data - self.final_yhat, np.sqrt(self.use_data))
        sr_hist, sr_bes = np.histogram(scaled_residual, bins=np.arange(-10,10,0.5))
        sr_bcs = sr_bes[1:] - 0.5*(sr_bes[1]-sr_bes[0])

        if plot:
            fig = plt.figure(constrained_layout=True, figsize=(18,6))
            w, h = [1, 1], [0.7, 0.3]
            spec = fig.add_gridspec(nrows=2, ncols=2, width_ratios=w, height_ratios=h) #, sharex=True)

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
