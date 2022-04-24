from copy import deepcopy
import dask.array as da
from datetime import datetime
import glob
import imageio as iio
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path, PurePath, PureWindowsPath
from scipy.interpolate import interp1d
from scipy.optimize import least_squares, leastsq
from scipy.signal import fftconvolve
import sys
sys.path.append(r"R:\OneDrive - Harvard University\lab-needleman\code\error_propagation-dev")
# from error_propagation import Complex


import warnings
import xarray as xr

# import time
from .util import *
from .sdt import *

if isnotebook():
    from tqdm.notebook import tqdm, trange
else:
    from tqdm import tqdm, trange


class decay_group:

    def __init__(self,
            data, irf, t_data=[], mask=None, fit_start_bin=None, fit_end_bin=None, modeltype="2exp", npx=1, manual_data=None, irf_kws={}
        ):

        self.t_irf        = irf["microtime_ns"].values
        self.dt_irf       = self.t_irf[1] - self.t_irf[0]
        self.dc_irf       = decay_curve(irf, plot=False, normalize=True, **irf_kws)

        self.nbins_irf    = len(self.t_irf)
        self.nphot_irf    = np.sum(self.dc_irf)
        self.dc_irf      /= (np.sum(self.dc_irf)*self.dt_irf)
        self.laser_period = irf.laser_period

        self.load_data(
            data,
            mask=mask,
            fit_start_bin=fit_start_bin,
            fit_end_bin=fit_end_bin,
            npx=npx,
            manual_data=manual_data
        )

        with np.errstate(divide="ignore"):
            # my weights
            self.fit_weight_sq = np.divide(1., self.use_data)[:,np.newaxis]
            self.fit_weight_sq[np.isinf(self.fit_weight_sq)] = 0
            self.fit_weight = np.sqrt(self.fit_weight_sq)

            # self.fit_weight_sq_arr = np.multiply(
            #         self.fit_weight_sq, #[:,np.newaxis],
            #         np.ones((1,self.nparams))
            #     )
        # Gavin / Tae weights
        # nonzero_use_data = self.use_data
        # nonzero_use_data[nonzero_use_data == 0] = 1
        # self.fit_weight_sq = np.divide(1., nonzero_use_data)[:,np.newaxis]
        # self.fit_weight = np.sqrt(self.fit_weight_sq)

    def get_param_values(self, values=None, errors=None):
        """
        Get estimates of parameter values and associated errors. Returns np.ndarray of shape (Nparam,2) where elements of each sub-list are value and error, respectively.
        """
        return np.array(
            [[self.params[iparam]["value"], self.params[iparam]["err"]] for iparam in self.params]
        )

    def set_param_value(self, name, value, error=0):
        self.params[name]["value"] = value
        self.params[name]["err"] = error

    def set_param_values(self, values=None, errors=None):
        """
        Assign parameter values and associated errors.
        """
        assert values.size == len(self.params)
        assert errors.size == len(self.params)

        for iparam, ival, ierr in zip([pname for pname in self.params], values, errors):
            self.params[iparam]["value"] = ival
            if errors is not None:
                self.params[iparam]["err"] = ierr

    def get_param_lims(self):
        """
        Get limits and step sizes for each parameter in model. Returns np.ndarray of shape (Nparam,3) where elements of each sub-list are lower bound, upper bound, and step size (for e.g. Gibbs sampling), respectively.
        """
        return np.array(
            [[self.params[iparam]["min"], self.params[iparam]["max"], self.params[iparam]["step"]] for iparam in self.params]
        )

    def load_data(self, data, mask=None, fit_start_bin=None, fit_end_bin=None, npx=1, manual_data=None):

        if (isinstance(data, np.ndarray) or isinstance(data, da.Array)) and len(data.shape) == 1:
            self.dc_data      = data
            if manual_data is not None:
                self.dc_data = manual_data
            self.nbins_data   = len(self.dc_data)
            self.adc_ratio    = self.nbins_irf//self.nbins_data
            self.t_data       = self.t_irf[(self.adc_ratio-1)::self.adc_ratio]
            self.dt_data      = self.dt_irf * len(self.dc_irf)/self.nbins_data
            self.nphot_data   = np.sum(self.dc_data)
            # self.intensity    = self.nphot_data / npx
        else:
            self.t_data       = data["microtime_ns"].values
            self.dt_data      = self.t_data[1] - self.t_data[0]
            self.dc_data      = decay_curve(data, plot=False, normalize=False, mask=mask)
            if manual_data is not None:
                self.dc_data = manual_data
            self.nbins_data   = len(self.t_data)
            self.adc_ratio    = self.nbins_irf//self.nbins_data
            self.nphot_data   = np.sum(self.dc_data)
            # self.intensity    = self.nphot_data / self.data.npx


        # select domain for fitting
        # default is to fit the region with zonzero values less a small buffer on either side
        # i.e. throw out the first and last 100 ps
        nonzero_data = np.argwhere(self.dc_data > 0)

        if len(nonzero_data) == 0: # no photons to use
            return False # failure

        if type(fit_start_bin) == int:
            self.fit_start_bin = fit_start_bin
        else:
            self.start_time = self.t_data[nonzero_data[0]] + 0.1
            self.fit_start_bin = np.argmin(np.abs(self.t_data - self.start_time))

        if type(fit_end_bin) == int:
            self.fit_end_bin = fit_end_bin
        else:
            self.end_time = self.t_data[nonzero_data[-1]] - 0.1
            self.fit_end_bin = np.argmin(np.abs(self.t_data - self.end_time))

        self.use_t          = self.t_data[self.fit_start_bin:(self.fit_end_bin+1)]
        self.use_data       = self.dc_data[self.fit_start_bin:(self.fit_end_bin+1)]
        self.nphot_in_range = np.sum(self.use_data)

        return True # success

    def est_A(self):
        """
        Estimate fraction of decay due to signal rather than noise
        """

        t_data_peak = self.t_data[np.argmax(self.dc_data)]

        t_noise_region_start = t_data_peak - 1.
        t_noise_region_end   = t_data_peak - 0.5
        # print(t_noise_region_start, t_noise_region_end)

        idx_noise_region_start = np.argmin(np.abs(self.t_data - t_noise_region_start))
        idx_noise_region_end   = np.argmin(np.abs(self.t_data - t_noise_region_end))

        mean_noise = np.mean(self.dc_data[idx_noise_region_start:(idx_noise_region_end+1)])

        return 1 - mean_noise/np.max(self.dc_data)

    def model_1exp(self, shift, A, tau1):

        # number of time bins (IRF time bin size) in a single laser period
        nbins_laser_period = int(self.laser_period/self.dt_irf)

        # time = one laser period
        t      = np.arange(0, self.laser_period, self.dt_irf)

        # values of model over this period
        model  = (1-A) + A*np.exp(-t/tau1)

        # duplicate model (add second period)
        dup_model = np.concatenate((model, model))

        # convolve model with IRF
        roll_irf  = np.roll(self.dc_irf, int(np.round(shift)))
        cnv       = fftconvolve(roll_irf, dup_model)

        # keep the second period of the convolution
        keep_start, keep_end = nbins_laser_period + 1, nbins_laser_period + 1 + self.nbins_irf
        p2        = cnv[keep_start:keep_end]

        # collapse down to the same number of bins as the data
        rshp      = np.sum(np.reshape(p2, (self.nbins_data, self.adc_ratio)), axis=1)

        # normalize to 1 within the time range of interest
        normed    = rshp/np.sum(rshp[self.fit_start_bin:(self.fit_end_bin+1)])

        # scale to number of photons within time range of interest, and truncate to this range
        return self.nphot_in_range*normed[self.fit_start_bin:(self.fit_end_bin+1)]

    def model_2exp(self, shift, A, tau1, tau2, f):

        # number of time bins (IRF time bin size) in a single laser period
        nbins_laser_period = int(self.laser_period/self.dt_irf)

        # time = one laser period
        t      = np.arange(0, self.laser_period, self.dt_irf)

        # values of model over this period
        model  = (1-A) + A*(f*np.exp(-t/tau1) + (1-f)*np.exp(-t/tau2))

        # duplicate model (add second period)
        dup_model = np.concatenate((model, model))

        # convolve model with IRF
        roll_irf  = np.roll(self.dc_irf, int(np.round(shift)))
        cnv       = fftconvolve(roll_irf, dup_model)

        # keep the second period of the convolution
        keep_start, keep_end = nbins_laser_period + 1, nbins_laser_period + 1 + self.nbins_irf
        p2        = cnv[keep_start:keep_end]

        # collapse down to the same number of bins as the data
        rshp      = np.sum(np.reshape(p2, (self.nbins_data, self.adc_ratio)), axis=1)

        # normalize to 1 within the time range of interest
        normed    = rshp/np.sum(rshp[self.fit_start_bin:(self.fit_end_bin+1)])

        # scale to number of photons within time range of interest, and truncate to this range
        return self.nphot_in_range*normed[self.fit_start_bin:(self.fit_end_bin+1)]

    def model_3exp(self, shift, A, tau1, tau2, tau3, f1, f2):

        # number of time bins (IRF time bin size) in a single laser period
        nbins_laser_period = int(self.laser_period/self.dt_irf)

        # time = one laser period
        t      = np.arange(0, self.laser_period, self.dt_irf)

        # values of model over this period
        model  = (1-A) + A*(f1*np.exp(-t/tau1) + (1-f1)*(f2*np.exp(-t/tau2) + (1-f2)*np.exp(-t/tau3)))

        # duplicate model (add second period)
        dup_model = np.concatenate((model, model))

        # convolve model with IRF
        roll_irf  = np.roll(self.dc_irf, int(np.round(shift)))
        cnv       = fftconvolve(roll_irf, dup_model)

        # keep the second period of the convolution
        keep_start, keep_end = nbins_laser_period + 1, nbins_laser_period + 1 + self.nbins_irf
        p2        = cnv[keep_start:keep_end]

        # collapse down to the same number of bins as the data
        rshp      = np.sum(np.reshape(p2, (self.nbins_data, self.adc_ratio)), axis=1)

        # normalize to 1 within the time range of interest
        normed    = rshp/np.sum(rshp[self.fit_start_bin:(self.fit_end_bin+1)])

        # scale to number of photons within time range of interest, and truncate to this range
        return self.nphot_in_range*normed[self.fit_start_bin:(self.fit_end_bin+1)]

    def model_4exp(self, shift, A, tau1, tau2, tau3, tau4, f1, f2, f3):

        # number of time bins (IRF time bin size) in a single laser period
        nbins_laser_period = int(self.laser_period/self.dt_irf)

        # time = one laser period
        t      = np.arange(0, self.laser_period, self.dt_irf)

        # values of model over this period
        model  = (1-A) + A*(f1*np.exp(-t/tau1) + (1-f1)*(f2*np.exp(-t/tau2) + (1-f2)*(f3*np.exp(-t/tau3) + (1-f3)*np.exp(-t/tau4))))

        # duplicate model (add second period)
        dup_model = np.concatenate((model, model))

        # convolve model with IRF
        roll_irf  = np.roll(self.dc_irf, int(np.round(shift)))
        cnv       = fftconvolve(roll_irf, dup_model)

        # keep the second period of the convolution
        keep_start, keep_end = nbins_laser_period + 1, nbins_laser_period + 1 + self.nbins_irf
        p2        = cnv[keep_start:keep_end]

        # collapse down to the same number of bins as the data
        rshp      = np.sum(np.reshape(p2, (self.nbins_data, self.adc_ratio)), axis=1)

        # normalize to 1 within the time range of interest
        normed    = rshp/np.sum(rshp[self.fit_start_bin:(self.fit_end_bin+1)])

        # scale to number of photons within time range of interest, and truncate to this range
        return self.nphot_in_range*normed[self.fit_start_bin:(self.fit_end_bin+1)]

    def jac(self, yhat0, model, p, dp=None):
        """

        yhat0: value of model at current parameters, should be equivalent to evaluating model(*p)
        model: model function
        p: parameter values around which we wish to calculate the Jacobian
        dp: step size used for calculating derivatives with respect to each parameter; same dimensions as p

        """

        # number of parameters
        nparams = len(p)

        # number of points at which function is to be evaluated
        npoints = len(yhat0)

        # if dp == None:
        #     dp = [1e-3 for ip in p]

        jac_val = np.zeros((npoints, nparams))

        # scale of change to each parameter
        dp_scale = dp*(1+np.abs(p))

        # calculate central difference for each parameter
        for iparam in range(nparams):

            if dp[iparam] == 0: continue

            p_change_iparam_fwd = p.copy()
            p_change_iparam_fwd[iparam] += dp_scale[iparam]

            p_change_iparam_rev = p.copy()
            p_change_iparam_rev[iparam] -= dp_scale[iparam]

            yhat_fwd = model(*p_change_iparam_fwd)
            yhat_rev = model(*p_change_iparam_rev)

            jac_val[:,iparam] = np.divide(yhat_fwd - yhat_rev, 2*dp_scale[iparam])

        return jac_val

    def lm_matx(self, data, model, p, dp=None):
        """
        returns:
        alpha = linearized Hessian matrix (inverse of covariance matrix)
        beta = linearized fitting vector
        X2 = 2*Chi squared criteria: weighted sum of the squared residuals WSSR
        y_hat = model evaluated with parameters 'p'
        dydp = derivative of yhat wrt parameters
        """
        # number of parameters
        nparams = len(p)

        # evaluate model at parameters p
        yhat = model(*p)
        residual = (data - yhat)[:, np.newaxis]

        # calculate Jacobian
        dydp = self.jac(data, model, p, dp)


        alpha = np.matmul(dydp.T, np.multiply(dydp, self.fit_weight_sq_arr))
        beta  = np.matmul(dydp.T, np.multiply(self.fit_weight_sq, residual))
        X2    = np.matmul(residual.T, np.multiply(residual, self.fit_weight_sq))[0,0]

        return alpha, beta, X2, yhat, dydp

    def custom_leastsq(
        self,
        data,
        model,
        p0,
        dp,
        pmin=None,
        pmax=None,
        maxit=1000,
        # eps1=1e-16,
        # eps2=1e-16,
        # eps3=1e-16,
        # eps4=1e-12,
        eps1=1e-8,
        eps2=1e-7,
        eps3=1e-6,
        eps4=1e-2,
        lambda_0=1e-2,
        lambda_up_fac=11,
        lambda_dn_fac=9,
        verbose=False
        ):

        status = 0
        stop = False

        dp = np.multiply(dp,
            np.array(np.array(pmin) != np.array(pmax), dtype=float)
        )

        p = p0
        alpha, beta, X2, yhat, dydp = self.lm_matx(data, model, p, dp=dp)
        residual = (data - yhat)[:, np.newaxis]

        if max(np.abs(beta)) < eps1:
            print(f"Initial guess is nearly optimal, so not continuing. eps1 = {eps1:4.2e}")
            stop = True

        lambda_val = lambda_0
        X2_old = X2

        # convergence history
        cvg_hst = []

        it = -1
        while (not stop) and (it <= maxit):
            it += 1

            deltap = np.squeeze(
                np.matmul(
                    np.linalg.pinv(alpha + lambda_val*np.diag(np.diag(alpha))),
                    beta
                )
            )

            ptry = p + deltap

            # enforce min/max constraints
            ptry = np.minimum(
                np.maximum(ptry, pmin),
                pmax
            )

            y1 = model(*ptry)
            residual = (data - y1)[:, np.newaxis]
            X2_try = np.matmul(residual.T, np.multiply(residual, self.fit_weight_sq))[0,0]
            rho = (X2 - X2_try) / np.dot(2*deltap, lambda_val*deltap + np.squeeze(beta))

            if rho > eps4: # p_try significantly better than p
                X2_old = X2
                pold = p.copy()
                p = ptry.copy()
                alpha, beta, X2, yhat, dydp = self.lm_matx(data, model, p, dp=dp)
                # decrease lambda (Gauss-Newton method)
                lambda_val = np.maximum(lambda_val/lambda_dn_fac, 1e-7)
            else: # not significantly better
                X2 = X2_old

                # increase lambda (gradient descent)
                lambda_val = np.minimum(lambda_val*lambda_up_fac, 1e7)

            cvg_hst.append((p, X2/2, lambda_val))

            if it > 1:
                if np.max(np.divide(deltap,p)) < eps2:
                    stop = True
                    if verbose:
                        print(f"Parameters converged; eps2={eps2:4.2e}")

                if X2/len(self.use_data) < eps3:
                    stop = True
                    if verbose:
                        print(f"X2 converged; eps3={eps3:4.2e}")

                if np.max(np.abs(beta)) < eps1:
                    stop = True
                    if verbose:
                        print(f"Beta converged; eps1={eps1:4.2e}")

            if it == maxit:
                warnings.warn("Maximum number of iterations reached without convergence.", UserWarning, stacklevel=4)
                status = 1


        """
        now estimate covariance / confidence intervals
        """

        # first, back up weights
        fit_weight_sq_old = self.fit_weight_sq.copy()
        fit_weight_old = self.fit_weight.copy()
        fit_weight_sq_arr_old = self.fit_weight_sq_arr.copy()

        # number of parameters, and number of points at which function is to be evaluated
        nparams, npoints = len(p0), len(data)
        nfitparams = np.sum(pmin == pmax)

        # now set equal weights for parameter error analysis
        self.fit_weight_sq = (npoints - nfitparams + 1)/np.dot(np.squeeze(residual), np.squeeze(residual)) * np.ones_like(data[:,np.newaxis])
        self.fit_weight = np.sqrt(self.fit_weight_sq)
        self.fit_weight_sq_arr = np.multiply(
                self.fit_weight_sq,
                np.ones((1,self.nparams))
            )

        alpha, beta, X2, yhat, dydp = self.lm_matx(data, model, p, dp=dp)

        covar = np.linalg.pinv(alpha)
        sigma_p = np.sqrt(np.diag(covar))

        # reset weights
        self.fit_weight_sq = fit_weight_sq_old.copy()
        self.fit_weight = fit_weight_old.copy()
        self.fit_weight_sq_arr = fit_weight_sq_arr_old.copy()

        return cvg_hst, p, sigma_p, status

    def log_likelihood_biexp(self, params, data=None):
        # if data == None:
        #     data = self.dc_data[self.fit_start_bin:(self.fit_end_bin+1)]

        return np.dot(
            np.log(self.model_fn(*params)),
            data
        )

    def posterior(self, param_mat, data=None, normalize=True):
        if data == None:
            data = self.dc_data[self.fit_start_bin:(self.fit_end_bin+1)]

        log_likelihood = np.apply_along_axis(lambda params: self.log_likelihood_biexp(params, data), 1, param_mat)
        likelihood = np.exp(log_likelihood - np.max(log_likelihood))

        if normalize:
            likelihood /= np.sum(likelihood)

        return likelihood

    def gibbs_sample(self,
            fix_p,
            nburn=50,
            nsample=100,
            verbose=False,
            showprogress=False
        ):

        # nuumber of burnin steps should be even
        if nburn%2 != 0: nburn += 1

        free_params = np.where( ~np.array(fix_p) )[0]

        param_lims = self.param_lims()
        param_min, param_max, param_step = param_lims[:,0], param_lims[:,1], param_lims[:,2]
        param_currvals = self.param_values()[:,0]
        param_currvals[0] = self.params["shift"]["value"]

        param_samples = np.empty((nsample, len(param_currvals)))
        burn_samples = np.empty((nburn//2, len(param_currvals)))

        if showprogress:
            iterator = trange(nburn+nsample)
        else:
            iterator = range(nburn+nsample)

        for it in iterator:
            for iparam in free_params:
                if it < nburn//2:
                    scale = 2
                else:
                    scale = 1

                iparam_range = np.arange(param_min[iparam], param_max[iparam], scale*param_step[iparam])

                # param_mat = matlib.repmat(param_currvals, len(iparam_range), 1)
                # need to test if below line using tile instead of repmat is correct
                param_mat = np.tile(param_currvals, (len(iparam_range), 1))
                param_mat[:,iparam] = iparam_range
                cond_post = self.posterior(param_mat)

                param_currvals[iparam] = np.random.choice(iparam_range, size=1, p=cond_post)

            if it >= nburn:
                param_samples[it-nburn,:] = param_currvals
            else:
                burn_samples[it-(nburn//2),:] = param_currvals

                if it == (nburn-1):

                    param_min = np.min(burn_samples, axis=0)
                    param_max = np.max(burn_samples, axis=0)

                    if np.all(np.abs(param_max - param_min) < 1e-6):
                        return burn_samples

                    for iparam in range(len(param_currvals)):
                        if param_min[iparam] == param_max[iparam]:
                            fix_p[iparam] = True
                    free_params = np.where( ~np.array(fix_p) )[0]

            if verbose:
                print(it, param_currvals)

        return param_samples

    def residual(self, data, fit_start_bin, fit_end_bin):
        return lambda p: np.multiply(data[fit_start_bin:fit_end_bin] - self.model_twoexp(data, p[0], p[1], p[2], p[3], p[4], fit_start_bin=fit_start_bin, fit_end_bin=fit_end_bin), self.fit_weight)

    def residual_fixshift(self, data, fit_start_bin, fit_end_bin, shift):
        return lambda p: np.multiply(data[fit_start_bin:fit_end_bin] - self.model_twoexp(data, shift, p[0], p[1], p[2], p[3], fit_start_bin=fit_start_bin, fit_end_bin=fit_end_bin), self.fit_weight)

    def fit(
            self,
            model="2exp",
            fixed_parameters=[],
            method="custom_leastsq",
            method_args={},
            leastsq_args={},
            save_leastsq_params_array=False,
            verbose=False,
            plot=False,
        ):

        if model == "1exp":

            self.model_fn = self.model_1exp

            self.params = {
                "shift": {"value": 0    , "err": np.nan, "min": -100 , "max":   100, "step": 1   },
                "A":     {"value": 0.995, "err": np.nan, "min": 0.700, "max": 1.000, "step": 1e-3},
                "tau1":  {"value": 3.500, "err": np.nan, "min": 2.000, "max": 5.000, "step": 1e-3},
            }

            fix_p = [False]*3
            for fp in fixed_parameters:
                if fp == "shift":
                    fix_p[0] = True
                elif fp == "A":
                    fix_p[1] = True
                elif fp == "tau1":
                    fix_p[2] = True
                else:
                    raise ValueError("Invalid parameter name.")

        elif model == "2exp":

            self.model_fn = self.model_2exp

            # self.params = {
            #     "shift": {"value": 0    , "err": np.nan, "min": -30  , "max":    30, "step": 1   },
            #     "A":     {"value": 0.995, "err": np.nan, "min": 0.700, "max": 1.000, "step": 1e-3},
            #     "tau1":  {"value": 3.500, "err": np.nan, "min": 2.000, "max": 5.000, "step": 1e-3},
            #     "tau2":  {"value": 1.000, "err": np.nan, "min": 0.010, "max": 0.800, "step": 1e-3},
            #     "f":     {"value": 0.405, "err": np.nan, "min": 0.010, "max": 1.000, "step": 1e-3}
            # }
            self.params = {
                "shift": {"value": 0    , "err": np.nan, "min": -200 , "max":   200, "step": 1   },
                "A":     {"value": 0.995, "err": np.nan, "min": 0.700, "max": 1.000, "step": 1e-3},
                "tau1":  {"value": 3.500, "err": np.nan, "min": 1.000, "max": 9.000, "step": 1e-3},
                "tau2":  {"value": 1.000, "err": np.nan, "min": 0.010, "max": 1.000, "step": 1e-3},
                "f":     {"value": 0.405, "err": np.nan, "min": 0.010, "max": 1.000, "step": 1e-3},
            }

            fix_p = [False]*5
            for fp in fixed_parameters:
                if fp == "shift":
                    fix_p[0] = True
                elif fp == "A":
                    fix_p[1] = True
                elif fp == "tau1":
                    fix_p[2] = True
                elif fp == "tau2":
                    fix_p[3] = True
                elif fp == "f":
                    fix_p[4] = True
                else:
                    raise ValueError("Invalid parameter name.")
        elif model == "3exp":

            self.model_fn = self.model_3exp

            self.params = {
                "shift": {"value": 0    , "err": np.nan, "min": -300 , "max":   300, "step": 1   },
                "A":     {"value": 0.995, "err": np.nan, "min": 0.700, "max": 1.000, "step": 1e-3},
                "tau1":  {"value": 3.500, "err": np.nan, "min": 1.000, "max": 9.000, "step": 1e-3},
                "tau2":  {"value": 0.500, "err": np.nan, "min": 0.010, "max": 1.000, "step": 1e-3},
                "tau3":  {"value": 0.500, "err": np.nan, "min": 0.010, "max": 1.000, "step": 1e-3},
                "f1":    {"value": 0.405, "err": np.nan, "min": 0.010, "max": 1.000, "step": 1e-3},
                "f2":    {"value": 0.405, "err": np.nan, "min": 0.010, "max": 1.000, "step": 1e-3},
            }

            fix_p = [False]*7
            for fp in fixed_parameters:
                if fp == "shift":
                    fix_p[0] = True
                elif fp == "A":
                    fix_p[1] = True
                elif fp == "tau1":
                    fix_p[2] = True
                elif fp == "tau2":
                    fix_p[3] = True
                elif fp == "tau3":
                    fix_p[4] = True
                elif fp == "f1":
                    fix_p[5] = True
                elif fp == "f2":
                    fix_p[6] = True
                else:
                    raise ValueError("Invalid parameter name.")
        elif model == "4exp":

            self.model_fn = self.model_4exp

            self.params = {
                "shift": {"value": 0    , "err": np.nan, "min": -300 , "max":   300, "step": 1   },
                "A":     {"value": 0.995, "err": np.nan, "min": 0.700, "max": 1.000, "step": 1e-3},
                "tau1":  {"value": 3.500, "err": np.nan, "min": 1.000, "max": 9.000, "step": 1e-3},
                "tau2":  {"value": 0.600, "err": np.nan, "min": 0.001, "max": 1.000, "step": 1e-3},
                "tau3":  {"value": 1.500, "err": np.nan, "min": 1.000, "max": 9.000, "step": 1e-3},
                "tau4":  {"value": 0.400, "err": np.nan, "min": 0.001, "max": 1.000, "step": 1e-3},
                "f1":    {"value": 0.200, "err": np.nan, "min": 0.001, "max": 1.000, "step": 1e-3},
                "f2":    {"value": 0.300, "err": np.nan, "min": 0.001, "max": 1.000, "step": 1e-3},
                "f3":    {"value": 0.500, "err": np.nan, "min": 0.001, "max": 1.000, "step": 1e-3},
            }

            fix_p = [False]*9
            for fp in fixed_parameters:
                if fp == "shift":
                    fix_p[0] = True
                elif fp == "A":
                    fix_p[1] = True
                elif fp == "tau1":
                    fix_p[2] = True
                elif fp == "tau2":
                    fix_p[3] = True
                elif fp == "tau3":
                    fix_p[4] = True
                elif fp == "tau4":
                    fix_p[5] = True
                elif fp == "f1":
                    fix_p[6] = True
                elif fp == "f2":
                    fix_p[7] = True
                elif fp == "f3":
                    fix_p[8] = True
                else:
                    raise ValueError("Invalid parameter name.")

        else:
            raise ValueError("Unrecognized model type.")

        self.nparams = len(self.params)

        #
        with np.errstate(divide="ignore"):
            self.fit_weight_sq = np.divide(1., self.use_data)[:,np.newaxis]
            self.fit_weight_sq[np.isinf(self.fit_weight_sq)] = 0
            self.fit_weight = np.sqrt(self.fit_weight_sq)

            self.fit_weight_sq_arr = np.multiply(
                    self.fit_weight_sq, #[:,np.newaxis],
                    np.ones((1,self.nparams))
                )

        if method == "custom_leastsq":

            lims = self.get_param_lims()
            pmin, pmax, dp = lims[:,0], lims[:,1], lims[:,2]
            init_params = np.mean(lims[:,0:2], axis=1)
            init_params[1] = self.est_A()

            for i, val in enumerate(fix_p):
                if (type(val) == float) or (type(val) == int):
                    init_params[i] = val
                    pmin[i] = val
                    pmax[i] = val
                elif (type(val) == bool) and val:
                    init_params[i] = self.params_leastsq[i]
                    pmin[i] = self.params_leastsq[i]
                    pmax[i] = self.params_leastsq[i]

            cvg_hst, p, sp, status = self.custom_leastsq(self.use_data, self.model_fn, init_params, dp, pmin=pmin, pmax=pmax, **method_args)

            if save_leastsq_params_array:
                self.params_leastsq = p

            self.set_param_values(values=p, errors=sp)

            # self.params["shift"]["value"], self.params["shift"]["err"]   = p[0], sp[0]
            # self.params["A"]["value"],     self.params["A"]["err"]       = p[1], sp[1]
            # self.params["tau1"]["value"],  self.params["tau1"]["err"]    = p[2], sp[2]
            # self.params["tau2"]["value"],  self.params["tau2"]["err"]    = p[3], sp[3]
            # self.params["f"]["value"],     self.params["f"]["err"]       = p[4], sp[4]
            # return p, sp
        # elif method == "sols":
        #     opt_result = leastsq(residual, [self.params["shift"], self.params["A"], self.params["tau1"], self.params["tau2"], self.params["f"]], epsfcn=[1, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])
        # elif method in ["leastsq", "emcee"]:
        #     if method == "leastsq":
        #         out = m.minimize(method=method, **leastsq_args)
        #     elif method == "emcee":
        #         out = m.minimize(method=method, **emcee_args)
        #     else:
        #         out = m.minimize(method=method, **method_args)
        #
        #     self.params = out.params.copy()

        if verbose:
            print(self.fit_params())

        if isinstance(plot, bool) and plot:
            self.plot()
        elif isinstance(plot, str):
            self.plot(path=plot)

        # return self.param_values()
        return pd.DataFrame(self.params).T, status


    def fit_params(self):
        return pd.DataFrame(self.params).T

    def plot(self,
            path="",
            params=None,
            datacolor="cornflowerblue",
            dataalpha=0.5,
            showfit=True,
            fitcolor="crimson",
            figsize=(12,4),
        ):

        if type(params) != np.ndarray:
            self.final_yhat = self.model_fn(*(self.get_param_values()[:,0]))
        else:
            self.final_yhat = self.model_fn(*params)

        scaled_residual = np.multiply(self.use_data - self.final_yhat, np.squeeze(self.fit_weight))

        res_bins = np.arange(-10,10,1)
        sr_hist, sr_bes = np.histogram(scaled_residual, bins=res_bins)
        sr_bcs = sr_bes[1:] - 0.5*(sr_bes[1]-sr_bes[0])

        fig = plt.figure(constrained_layout=True, figsize=figsize)
        plt.suptitle(f"N = {int(self.dc_data.sum()):5d} photons")
        w, h = [1, 1], [0.7, 0.3]
        spec = fig.add_gridspec(nrows=2, ncols=2, width_ratios=w, height_ratios=h) #, sharex=True)

        xlims = [-0.5,12.5]
        ax = fig.add_subplot(spec[0,0])
        ax.plot(self.t_data, self.dc_data, "o", color=datacolor, alpha=dataalpha, label="data")
        if showfit:
            ax.plot(self.use_t, self.final_yhat, color=fitcolor, linestyle="-", label="fit")
            ax2 = ax.twinx()
            ax2.plot(self.t_irf, self.dc_irf, color="olivedrab", alpha=0.85, label="IRF")
        ax.set_xlim(xlims)
        ax.set_xticks(np.arange(0,12.6))
        ax.legend()
        ax.set_xlabel(r"$t$/ns")
        ax.set_ylabel(r"counts")
        ax.set_yscale("log")
        ax2.set_yscale("log")

        ax = fig.add_subplot(spec[0,1])
        ax.plot(self.t_data, self.dc_data, "o", color=datacolor, alpha=dataalpha, label="data")
        if showfit:
            ax.plot(self.use_t, self.final_yhat, color=fitcolor, linestyle="-", label="fit")
            ax2 = ax.twinx()
            ax2.plot(self.t_irf, self.dc_irf, color="olivedrab", alpha=0.85, label="IRF")
        ax.set_xlim(xlims)
        ax.set_xticks(np.arange(0,12.6))
        ax.set_xlabel(r"$t$/ns")

        ax = fig.add_subplot(spec[1,0])
        ax.plot(self.use_t, scaled_residual, color=datacolor, alpha=dataalpha*2)
        ax.set_xlim(xlims)
        ax.set_xticks(np.arange(0,12.6))
        ax.axhline(y=0, color="gray")
        # ax.set_ylim([-0.05,0.05])
        ax.set_xlabel(r"$t$/ns")
        ax.set_ylabel(r"scaled residual")

        ax = fig.add_subplot(spec[1,1])
        ax.bar(sr_bcs, sr_hist, width=sr_bcs[1]-sr_bcs[0], color=datacolor, alpha=dataalpha)
        ax.set_xticks(res_bins)
        ax.set_xlabel(r"scaled residual")
        ax.set_ylabel(r"counts")

        if len(path) > 0:
            plt.savefig(path, dpi=300)
            plt.close(fig)
        else:
            plt.show()

def calculate_FLIM_fits(
    da_flim,
    da_segmentation,
    irf,
    irf_kws=dict(),
    parameter_vals=["shift", "A", "tau1", "tau2", "f"],
    value_type_vals=["value", "err", "min", "max", "step"]
):

    def single_flim_fit(curr_flim_image, curr_mask):

        dg = decay_group(
            decay_curve(curr_flim_image.data, mask=curr_mask.data),
            irf.sel(channel="M2"),
            irf_kws=irf_kws,
        )

        fit_result = dg.fit()

        if fit_result[1] == 0:
            return fit_result[0].values
        else:
            return np.nan*fit_result[0].values

    parameter_type_vals = ["_".join(iprod) for iprod in itertools.product(parameter_vals, value_type_vals)]

    return xr.apply_ufunc(
        single_flim_fit,
        da_flim,
        da_segmentation,
        input_core_dims=[["y", "x", "microtime_ns"], ["y", "x"]],
        output_core_dims=[["parameter", "value_type"]],
        output_dtypes=np.float32,
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs=dict(allow_rechunk=True, output_sizes={"parameter": len(parameter_vals), "value_type": len(value_type_vals)},),
    ).assign_coords(parameter_type=(["parameter", "value_type"], np.array(parameter_type_vals).reshape((len(parameter_vals),len(value_type_vals)))))
    # ).assign_coords(parameter=parameter_vals, value_type=value_type_vals)

    # return xr.apply_ufunc(
    #     single_flim_fit,
    #     da_flim,
    #     da_segmentation,
    #     input_core_dims=[["y", "x", "microtime_ns"], ["y", "x"]],
    #     output_core_dims=[["parameter", "value_type"]],
    #     output_sizes={"parameter": 5, "value_type": 5},
    #     output_dtypes=np.float32,
    #     vectorize=True,
    #     dask="parallelized",
    #     dask_gufunc_kwargs={"allow_rechunk": True},
    # )

def FLIM_fits_xda_to_df(da_flim_fits, existing_index=["timepoint", "position",]):

    df = da_flim_fits.to_dataframe("parameter_value")

    df = pd.pivot(
        data=df.reset_index().drop(["parameter", "value_type"], axis=1),
        values="parameter_value",
        index=existing_index + ["filename",],
        columns=["parameter_type"]
    )

    return df

def NADHFLIM_beta(f):
    return f/(1-f)

def NADHFLIM_CNADHfree(intensity, taushort, taulong, f, scale_factor=1):
    return intensity*(1-f)/(scale_factor*((taulong-taushort)*f + taushort))

def NADHFLIM_CNADHfree_df(df, scale_factor=1):
    df = df.copy()
    df["CNADHf"] = df["intensity"]*(1-df["f"])/(scale_factor*((df["tau1"]-df["tau2"])*df["f"] + df["tau2"]))
    return df

def NADHFLIM_rox(f, feq, alpha=1):
    beta = NADHFLIM_beta(f)
    betaeq = NADHFLIM_beta(feq)
    return alpha*(beta-betaeq)

def NADHFLIM_rox_df(df, is_eq, alpha=1):

    df = df.copy()
    df["beta"] = NADHFLIM_beta(df["f"])
    # beta = NADHFLIM_beta(df.loc[np.logical_not(is_eq),"f"])
    betaeq = np.mean(NADHFLIM_beta(df.loc[is_eq,"f"]))
    df["rox"] = alpha*(df["beta"]-betaeq)
    return df

def NADHFLIM_JETC(intensity, taushort, taulong, f, feq, scale_factor=1, alpha=1):
    """
    Calculate ETC flux using NADH FLIM parameters (Xingbo's method)
    """

    return NADHFLIM_rox(f,feq,alpha)*NADHFLIM_CNADHfree(intensity, taushort, taulong, f, scale_factor)

def NADHFLIM_JETC_df(df, is_eq, scale_factor=1, alpha=1):
    """
    Calculate ETC flux using NADH FLIM parameters (Xingbo's method)
    """
    df = df.copy()
    df = NADHFLIM_CNADHfree_df(df, scale_factor)
    df = NADHFLIM_rox_df(df, is_eq, alpha)
    df["JFLIM"] = df["rox"] * df["CNADHf"]

    return df
