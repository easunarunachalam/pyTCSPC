__all__ = [
    "calc_acf",
]

import numpy as np
import xarray as xr
from multipletau import autocorrelate

def calc_acf(nc_fn, dt=1e-6, m=16, maxtime=None, maxtime_subtraj=None):
    all_photons = xr.load_dataset( nc_fn )
    
    # macrotimes, in units of seconds
    times = all_photons["time"].data
    
    if maxtime is None:
        maxtime = np.max(times)
    if maxtime_subtraj is None:
        maxtime_subtraj = maxtime

    cumulative_acf = None
    nphotons = 0
    for j in np.arange(0,maxtime,maxtime_subtraj):
            
        curr_photons = all_photons.sel(time=slice(j, j+maxtime_subtraj))
        curr_times = curr_photons["time"].data
        bin_edges = np.arange(j,j+maxtime_subtraj,dt)
        counts, bin_edges = np.histogram(curr_times, bins=bin_edges)
        lag_time, acf = tuple(autocorrelate(counts.astype(float), normalize=True, deltat=dt, m=m).T)
        # lag_time, acf = tuple(autocorrelate(counts.astype(float), subtract_mean=True, normalize_to_mean=True, normalize_by_bin_factor=True, deltat=dt, m=m).T)

        curr_nphotons = len(curr_times)

        if cumulative_acf is None:
            cumulative_acf = acf*curr_nphotons
        else:
            cumulative_acf += acf*curr_nphotons
        nphotons += curr_nphotons

    avg_acf = cumulative_acf / nphotons

    return lag_time[1:], avg_acf[1:], nphotons