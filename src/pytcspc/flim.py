__all__ = [
    "resample_from_hist",
    "decay_group",
    "amplitude_distribution"
]

from corner import corner
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import PurePath
from scipy.signal import fftconvolve

import warnings

from .presets import *
from .util import *
from .sdt import *
from .sdt import decay_curve

if isnotebook():
    from tqdm.notebook import trange
else:
    from tqdm import trange

def resample_from_hist(h, n=1):
    """
    Given a histogram `h`, subsample `n` values and construct a histogram of these samples
    """
    x = np.arange(len(h))
    h_norm = h / np.sum(h)

    samples = np.random.choice(x, size=(n,), p=h_norm)

    bin_edges = np.arange(0, len(h)+1) - 0.5
    h_samples, _ = np.histogram(samples, bins=bin_edges)

    return h_samples


class decay_group:

    def __init__(self,
            data, irf, t_data=[], mask=None, fit_start_bin=None, fit_end_bin=None, fit_decaymask=None, refcurve=None, npx=1, manual_data=None, irf_kws={}
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
            refcurve=refcurve,
            npx=npx,
            manual_data=manual_data
        )

        with np.errstate(divide="ignore"):
            # my weights
            self.fit_weight_sq = np.divide(1., self.use_data)[:,np.newaxis]
            if fit_decaymask is not None:
                fit_decaymask = fit_decaymask[self.fit_start_bin:(self.fit_end_bin+1)]
                self.fit_weight_sq = np.multiply(self.fit_weight_sq, fit_decaymask)
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

    def set_param_samples(self, param_samples):
        """
        Assign sampled values for parameters (e.g. from Gibbs sampling).
        """
        self.param_samples = param_samples

    def get_param_lims(self):
        """
        Get limits and step sizes for each parameter in model. Returns np.ndarray of shape (Nparam,3) where elements of each sub-list are lower bound, upper bound, and step size (for e.g. Gibbs sampling), respectively.
        """
        return np.array(
            [[self.params[iparam]["min"], self.params[iparam]["max"], self.params[iparam]["step"]] for iparam in self.params]
        )

    def load_data(self, data, mask=None, fit_start_bin=None, fit_end_bin=None, refcurve=None, npx=1, manual_data=None):

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

        # load reference curve
        if isinstance(refcurve, np.ndarray):
            self.refcurve = refcurve
        elif isinstance(refcurve, str) or isinstance(refcurve, PurePath):
            self.refcurve = np.loadtxt(refcurve)
        else:
            self.refcurve = np.ones_like(self.dc_data)

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
        self.use_refcurve   = self.refcurve[self.fit_start_bin:(self.fit_end_bin+1)]
        self.nphot_in_range = np.sum(self.use_data)

        return True # success

    def est_A(self):
        """
        Estimate fraction of decay due to signal rather than background
        """

        t_data_peak = self.t_data[np.argmax(self.dc_data)]

        t_noise_region_start = t_data_peak - 1.
        t_noise_region_end   = t_data_peak - 0.5
        # print(t_noise_region_start, t_noise_region_end)

        idx_noise_region_start = np.argmin(np.abs(self.t_data - t_noise_region_start))
        idx_noise_region_end   = np.argmin(np.abs(self.t_data - t_noise_region_end))

        mean_noise = np.mean(self.dc_data[idx_noise_region_start:(idx_noise_region_end+1)])

        return 1 - mean_noise/np.max(self.dc_data)
    
    def rawmodel_to_fullmodel(self, rawmodel, shift):
        """
        Given a single-exponential or sum-of-exponentials model, compute the convolution with the
        instrument response function, normalize, and correct using the reference curve to prepare
        it to use as a full fitting function.
        """

        # number of time bins (IRF time bin size) in a single laser period
        nbins_laser_period = int(self.laser_period/self.dt_irf)

        # duplicate model (add second period)
        dup_model = np.concatenate((rawmodel, rawmodel))

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

        # scale to number of photons within time range of interest, truncate to this range, and scale by reference curve
        return self.use_refcurve * self.nphot_in_range * normed[self.fit_start_bin:(self.fit_end_bin+1)]

    def model_1exp(self, shift, A, tau1):

        # time = one laser period
        t      = np.arange(0, self.laser_period, self.dt_irf)

        # values of model over this period
        model  = (1-A) + A*np.exp(-t/tau1)

        return self.rawmodel_to_fullmodel(model, shift)

    def model_2exp(self, shift, A, tau1, tau2, f):

        # time = one laser period
        t      = np.arange(0, self.laser_period, self.dt_irf)

        # values of model over this period
        model  = (1-A) + A*(f*np.exp(-t/tau1) + (1-f)*np.exp(-t/tau2))

        return self.rawmodel_to_fullmodel(model, shift)

    def model_3exp(self, shift, A, tau1, tau2, tau3, f1, f2):

        # time = one laser period
        t      = np.arange(0, self.laser_period, self.dt_irf)

        # values of model over this period
        model  = (1-A) + A*(f1*np.exp(-t/tau1) + (1-f1)*(f2*np.exp(-t/tau2) + (1-f2)*np.exp(-t/tau3)))

        return self.rawmodel_to_fullmodel(model, shift)

    def model_4exp(self, shift, A, tau1, tau2, tau3, tau4, f1, f2, f3):

        # time = one laser period
        t      = np.arange(0, self.laser_period, self.dt_irf)

        # values of model over this period
        model  = (1-A) + A*(f1*np.exp(-t/tau1) + (1-f1)*(f2*np.exp(-t/tau2) + (1-f2)*(f3*np.exp(-t/tau3) + (1-f3)*np.exp(-t/tau4))))

        return self.rawmodel_to_fullmodel(model, shift)

    def model_polyexp(
        self, shift, A, *args, min_tau=0.01, max_tau=5.0, n_tau=200
    ):
        """
        Sum of many exponentials, with weights given by a sum of gaussians with amplitudes in `gaussian_amp`, means in `gaussian_mus`, and standard deviations in `gaussian_sigmas`
        """

        # read in parameters of gaussians
        assert (len(args)+1) % 3 == 0
        n_gaussians = (len(args)+1) // 3

        gaussian_amps   = np.full((n_gaussians,), np.nan)
        gaussian_mus    = np.full((n_gaussians,), np.nan)
        gaussian_sigmas = np.full((n_gaussians,), np.nan)

        for i, iarg in enumerate(np.arange(0,2*n_gaussians,2)):
            gaussian_mus[i]    = args[iarg]
            gaussian_sigmas[i] = args[iarg+1]

        for i, iarg in enumerate(np.arange(2*n_gaussians,len(args),1)):
            gaussian_amps[i] = args[iarg]

        # enforce amplitude sum = 1
        gaussian_amps = np.clip(gaussian_amps, 0.0, 1.0)
        gaussian_amps[-1] = 1. - np.sum(gaussian_amps[:-1])

        log_min_tau, log_max_tau = np.log10(min_tau), np.log10(max_tau)
        log_taus = np.linspace(log_min_tau, log_max_tau, n_tau)
        taus = 10**log_taus

        t = np.arange(0, self.laser_period, self.dt_irf)
        tile_taus = np.tile(taus, (len(t),1))
        tile_microtimes = np.tile(t, (len(taus), 1)).T
        exps = np.exp(-np.divide(tile_microtimes,tile_taus))

        exp_weights = amplitude_distribution(taus, gaussian_amps, gaussian_mus, gaussian_sigmas)
        tile_exp_weights = np.tile(exp_weights, (len(t), 1))

        exps_only = np.multiply(tile_exp_weights, exps).sum(axis=1)

        model = (1-A) + A*exps_only

        return self.rawmodel_to_fullmodel(model, shift)

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

    def log_likelihood(self, params, data=None):
        # if data == None:
        #     data = self.dc_data[self.fit_start_bin:(self.fit_end_bin+1)]

        return np.dot(
            np.log(self.model_fn(*params)),
            data
        )

    def posterior(self, param_mat, data=None, normalize=True):
        if data == None:
            data = self.dc_data[self.fit_start_bin:(self.fit_end_bin+1)]

        log_likelihood = np.apply_along_axis(lambda params: self.log_likelihood(params, data), 1, param_mat)
        likelihood = np.exp(log_likelihood - np.max(log_likelihood))

        if normalize:
            likelihood /= np.sum(likelihood)

        return likelihood

    def gibbs_sample(self,
            fix_p,
            nburn=50,
            nsample=10,
            verbose=False,
            show_progress=False
        ):

        # number of burnin steps should be even
        if nburn%2 != 0: nburn += 1

        free_params = np.where( ~np.array(fix_p) )[0]

        param_lims = self.get_param_lims()
        param_min, param_max, param_step = param_lims[:,0], param_lims[:,1], param_lims[:,2]
        param_currvals = self.get_param_values()[:,0]
        param_currvals[0] = self.params["shift"]["value"]

        param_samples = np.empty((nsample, len(param_currvals)))
        burn_samples = np.empty((nburn//2, len(param_currvals)))

        if show_progress:
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
                # print("param mat", param_mat)
                cond_post = self.posterior(param_mat)
                # print("cond post", cond_post)

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
            parameters=None,
            fixed_parameters=[],
            method="custom_leastsq",
            method_args={},
            leastsq_args={},
            save_leastsq_params_array=False,
            verbose=False,
            plot=False,
        ):
        """
        Fit a model to loaded data.
        Specify initial parameters for the fit via `parameters`, and include any fixed parameters in the list `fixed_parameters` as a string. Pass any optional arguments to specific fitting method via `method_args`.
        If plot=True, show the result of the fit on top of the data.
        """

        if model == "1exp":
            self.model_fn = self.model_1exp

            if parameters is None:
                self.params = {
                    "shift": {"value": 0    , "err": np.nan, "min": -200 , "max":   200, "step": 1   },
                    "A":     {"value": 0.995, "err": np.nan, "min": 0.700, "max": 1.000, "step": 1e-3},
                    "tau1":  {"value": 3.500, "err": np.nan, "min": 0.100, "max": 5.000, "step": 1e-3},
                }
            else:
                self.params = reorder_dict(parameters, ["shift", "A", "tau1"])

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

            if parameters is None:
                self.params = NADH_2EXP_INIT
            else:
                self.params = reorder_dict(parameters, ["shift", "A", "tau1", "tau2", "f"])

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

            if parameters is None:
                self.params = {
                    "shift": {"value": 0    , "err": np.nan, "min": -300 , "max":   300, "step": 1   },
                    "A":     {"value": 0.99, "err": np.nan, "min": 0.700, "max": 1.000, "step": 1e-3},
                    "tau1":  {"value": 3.500, "err": np.nan, "min": 1.000, "max": 9.000, "step": 1e-3},
                    "tau2":  {"value": 0.500, "err": np.nan, "min": 0.010, "max": 1.000, "step": 1e-3},
                    "tau3":  {"value": 0.200, "err": np.nan, "min": 0.010, "max": 1.000, "step": 1e-3},
                    "f1":    {"value": 0.15, "err": np.nan, "min": 0.010, "max": 1.000, "step": 1e-3},
                    "f2":    {"value": 0.40, "err": np.nan, "min": 0.010, "max": 1.000, "step": 1e-3},
                }
            else:
                self.params = reorder_dict(parameters, ["shift", "A", "tau1", "tau2", "tau3", "f1", "f2"])

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

            if parameters is None:
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
            else:
                self.params = reorder_dict(parameters, ["shift", "A", "tau1", "tau2", "tau3", "tau4", "f1", "f2", "f3"])

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
                
        elif model == "polyexp":

            self.model_fn = self.model_polyexp

            if parameters is None:
                self.params = {
                    "shift":  {"value": 0    , "err": np.nan, "min": -300 , "max":   300, "step": 1   },
                    "A":      {"value": 0.995, "err": np.nan, "min": 0.700, "max": 1.000, "step": 1e-3},
                    "mu1":    {"value": 0.2, "err": np.nan, "min": 0.001, "max": 10.0, "step": 1e-3},
                    "sigma1": {"value": 0.1, "err": np.nan, "min": 0.001, "max": 1.00, "step": 1e-3},
                    "mu2":    {"value": 2.0, "err": np.nan, "min": 0.001, "max": 10.0, "step": 1e-3},
                    "sigma2": {"value": 0.5, "err": np.nan, "min": 0.001, "max": 1.00, "step": 1e-3},
                    "amp1":   {"value": 0.8, "err": np.nan, "min": 0.000, "max": 1.00, "step": 1e-3},
                }
            else:
                self.params = reorder_dict(parameters, ["shift", "A", "mu1", "sigma1", "mu2", "sigma2", "amp1"])

            fix_p = [False]*len(self.params)
            # for fp in fixed_parameters:
            #     if fp == "shift":
            #         fix_p[0] = True
            #     elif fp == "A":
            #         fix_p[1] = True
            #     elif fp == "amp1":
            #         fix_p[2] = True
            #     elif fp == "mu1":
            #         fix_p[3] = True
            #     elif fp == "sigma1":
            #         fix_p[4] = True
            #     elif fp == "amp2":
            #         fix_p[5] = True
            #     elif fp == "mu2":
            #         fix_p[6] = True
            #     elif fp == "sigma2":
            #         fix_p[7] = True
            #     else:
            #         raise ValueError("Invalid parameter name.")

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
            # init_params = np.mean(lims[:,0:2], axis=1)
            init_params = self.get_param_values()[:,0]
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

            ret_val = pd.DataFrame(self.params).T, status

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

        elif method == "gibbs_sample":
            param_samples = self.gibbs_sample(fix_p, **method_args)
            
            df_param_samples = pd.DataFrame(columns=list(self.params), data=param_samples)

            ci = 95
            ci_hi = np.percentile(param_samples, (100+ci)/2., axis=0)
            ci_lo = np.percentile(param_samples, (100-ci)/2., axis=0)
            p = np.median(param_samples, axis=0)
            sp = (ci_hi-ci_lo)/2.35
            self.set_param_values(values=p, errors=sp)
            self.set_param_samples(df_param_samples)

            ret_val = pd.DataFrame(self.params).T, df_param_samples
        else:
            print("")
            raise ValueError("Unknown fitting method specified")

        if verbose:
            print(self.fit_params())

        if isinstance(plot, bool) and plot:
            self.plot()
        elif isinstance(plot, str):
            self.plot(path=plot)

        # return self.param_values()
        return ret_val


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
            lw=None, ms=None
            # figsize=(6,2),
            # lw=0.75, ms=2
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
        ax.plot(self.t_data, self.dc_data, "o", ms=ms, color=datacolor, alpha=dataalpha, label="data")
        if showfit:
            ax.plot(self.use_t, self.final_yhat, lw=lw, color=fitcolor, linestyle="-", label="fit")
            ax2 = ax.twinx()
            ax2.plot(self.t_irf, self.dc_irf, lw=lw, color="olivedrab", alpha=0.85, label="IRF")
        ax.set_xlim(xlims)
        ax.set_xticks(np.arange(0,12.6))
        ax.legend()
        ax.set_xlabel(r"$t$/ns")
        ax.set_ylabel(r"counts")
        ax.set_yscale("log")
        ax2.set_yscale("log")

        ax = fig.add_subplot(spec[0,1])
        ax.plot(self.t_data, self.dc_data, "o", ms=ms, color=datacolor, alpha=dataalpha, label="data")
        if showfit:
            ax.plot(self.use_t, self.final_yhat, lw=lw, color=fitcolor, linestyle="-", label="fit")
            ax2 = ax.twinx()
            ax2.plot(self.t_irf, self.dc_irf, lw=lw, color="olivedrab", alpha=0.85, label="IRF")
        ax.set_xlim(xlims)
        ax.set_xticks(np.arange(0,12.6))
        ax.set_xlabel(r"$t$/ns")

        ax = fig.add_subplot(spec[1,0])
        ax.plot(self.use_t, scaled_residual, lw=lw, color=datacolor, alpha=dataalpha*2)
        ax.set_xlim(xlims)
        ax.set_xticks(np.arange(0,12.6))
        ax.axhline(y=0, color="gray")
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

    def plot_samples(self):
        """
        Make a corner plot using parameter samples, e.g. from Gibbs sampling
        """
        range_is_zero = (np.max(self.param_samples, axis=0) - np.min(self.param_samples, axis=0)) == 0
        columns_to_drop = range_is_zero.loc[range_is_zero].index.values
        param_samples_to_plot = self.param_samples.drop(columns_to_drop, axis=1)

        labels = list(param_samples_to_plot.columns)
        for i in range(len(labels)):
            if labels[i][:3] == "tau":
                labels[i] = r"$\tau" + "_{" + labels[i][3:] + "}$/ns"
            else:
                labels[i] = r"$" + labels[i] + r"$"

        # fig, ax = plt.subplots(figsize=())
        corner(param_samples_to_plot, labels=labels)
        plt.show()

def _gaussian(x, mu, sigma):
    """
    Normal distribution
    use for polyexp fitting
    """
    norm_factor = (1./np.sqrt(2*np.pi)) * (1/sigma)
    return norm_factor * np.exp(-((x-mu)/sigma)**2)

def amplitude_distribution(taus, gaussian_amps, gaussian_mus, gaussian_sigmas):
    """
    Construct sum-of-gaussians amplitude distribution given parameters of gaussians
    """
    exp_weights_from_each_gaussian = np.array([gaussian_amp*_gaussian(taus, gaussian_mu, gaussian_sigma) for gaussian_amp, gaussian_mu, gaussian_sigma in zip(gaussian_amps, gaussian_mus, gaussian_sigmas)])

    return np.sum(exp_weights_from_each_gaussian, axis=0)