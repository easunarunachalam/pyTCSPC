from datetime import datetime
import glob
import h5py
import imageio as iio
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import os
from pathlib import Path, PurePath, PureWindowsPath
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve, convolve
import sys
sys.path.append(r"R:\OneDrive - Harvard University\lab-needleman\code\error_propagation-dev")
from typing import Union
import time

# from error_propagation import Complex


if "xarray" in sys.modules:
    import xarray as xr

import imageio
import matplotlib.pyplot as plt
import numpy as np

from skimage.filters import rank, gaussian
from skimage.segmentation import watershed
from skimage.morphology import disk
from skimage.transform import rescale

def isnotebook():
    """
    Determine whether current shell is a notebook or standard interpreter
    From https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True   # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

if isnotebook():
    from tqdm.notebook import tqdm, trange
else:
    from tqdm import tqdm, trange

# from util import colorbar

def contains_any_targets(test, targets):
    return np.any([(target in test) for target in targets])

def dirs_fns_to_process(glob_str, exclude_in_names=["calibration", "zarr"]):
    sdt_file_list = np.array([Path(fp) for fp in glob.glob(glob_str, recursive=True) if not contains_any_targets(str(fp), exclude_in_names)])
    sdt_file_dirs = np.array([fn.parent for fn in sdt_file_list])
    sdt_file_dirs_use, invs = np.unique([fn for fn in sdt_file_dirs], return_inverse=True)
    sdt_file_lists = [sdt_file_list[invs == iunq] for iunq in range(len(sdt_file_dirs_use))]

    return sdt_file_dirs_use, sdt_file_lists

def h5_to_dict(path):
    with h5py.File(path, "r") as hf:
        return { key: np.array(hf[key]) for key in list(hf.keys()) }

# frame = line = pixel = np.random.rand((10000))
#
# h5_filepath = Path("test.h5")
#
# with h5py.File(h5_filepath, "w") as hf:
#     hf.create_dataset("raw/frame", data=frame)
#     hf.create_dataset("raw/line", data=line)
#     hf.create_dataset("raw/pixel", data=pixel)
#     hf.create_dataset("raw/microtimel", data=pixel)
#
#     hf.create_dataset("traces/frame_time", data=frame_time)
#     hf.create_dataset("traces/intensity", data=intensity)
#     hf.create_dataset("traces/sum_lifetime", data=sum_lifetime)

def print_h5(h5_obj, indent_level=0, indent_str="  "):
    """
    Recursively print contents of h5 file in tree structure
    """
    if hasattr(h5_obj, "keys"):
        print(indent_str*indent_level, h5_obj)
        for key in list(h5_obj.keys()):
            print_h5(h5_obj[key], indent_level+1)
    else:
        print(indent_str*indent_level, h5_obj)

# with h5py.File(h5_filepath, "r") as hf:
#     print_h5(hf, 0)

def filewrite_mode(path):
    """
    Return mode for writing to file -- write ("w") if file does not yet exist, or append ("a") if file already exists
    """
    if path.is_file():
        return "a"
    else:
        return "w"

def within_percentile(array, percentile_lo, percentile_hi):
    return array[np.logical_and(
        array > np.percentile(array, percentile_lo),
        array < np.percentile(array, percentile_hi)
    )]

def _zoom_image(
        img:np.ndarray,
        zoom_factor:Union[int,float]=1
    ):
    """
    Zoom in on images while preserving resolution.
    """

    if not (isinstance(zoom_factor,int) or isinstance(zoom_factor,float)):
        raise TypeError(f"Invalid value for zoom factor '{zoom_factor}'. Must be int or float.")

    if zoom_factor == 1:
        return img
    else:
        # zoom in on center of image
        sz = np.array(img.shape, dtype=int)
        new_sz = int(sz/zoom_factor)
        rem_sz = np.array((sz - new_sz)/2, dtype=int)
        img = img[rem_sz[0]:-rem_sz[0],rem_sz[1]:-rem_sz[1]]

        return img

def _intensity_correction_mask_ndarray(
        intensity_illprof:np.ndarray,
        zoom_factor:Union[float,int]=1,
        rescale_factor:Union[float,int]=1,
        sigma:Union[float,int]=15,
        diskr:Union[float,int]=6,
    ):
    """
    Given an illumination profile, return a smoothed inverse image which can be used to correct intensity images for inhomogeneous illumination.
    """

    intensity_illprof = _zoom_image(intensity_illprof, zoom_factor=zoom_factor)

    if rescale_factor is not None:
        intensity_illprof = rescale(intensity_illprof, rescale_factor, anti_aliasing=False)

    # last row is sometimes blank, so we will ignore it when smoothing
    ignore_last_n = -1
    ip_img_crop = intensity_illprof[:ignore_last_n,:]

    # smooth image
    sm_ip_img = gaussian(ip_img_crop, sigma=10, preserve_range=True)

    # invert image
    boost_mask = intensity_illprof.astype(np.float32)
    boost_mask[:ignore_last_n,:] = np.max(sm_ip_img)/sm_ip_img

    return boost_mask

def intensity_correction_mask(
        intensity_illprof:Union[xr.DataArray,np.ndarray],
        zoom_factor:Union[float,int]=1,
        rescale_factor:Union[float,int]=1,
        sigma:Union[float,int]=15,
        diskr:Union[float,int]=6,
    ):
    """
    Given an illumination profile, return a smoothed inverse image which can be used to correct intensity images for inhomogeneous illumination.
    """

    if isinstance(intensity_illprof, xr.DataArray):
        boost_mask = intensity_illprof.copy(deep=True)
        for ichannel in boost_mask.channel:
            ip_slice = boost_mask.sel(channel=ichannel).data
            boost_mask.loc[dict(channel=ichannel)] = _intensity_correction_mask_ndarray(
                    ip_slice,
                    zoom_factor,
                    rescale_factor,
                    sigma,
                    diskr,
                )
        return boost_mask
    else:
        boost_mask = _intensity_correction_mask_ndarray(
                intensity_illprof,
                zoom_factor,
                rescale_factor,
                sigma,
                diskr,
            )
from scipy import interpolate

def resample_image(orig_img, new_nx, new_ny, interp_method="linear"):
    """
    Upsample or downsample a 2d image to target number of pixels along x and y axes
    """

    orig_x = np.arange(np.shape(orig_img)[0]) / np.shape(orig_img)[0]
    orig_y = np.arange(np.shape(orig_img)[1]) / np.shape(orig_img)[1]
    f = interpolate.interp2d(orig_x, orig_y, orig_img, kind=interp_method)
    new_x, new_y = np.arange(new_nx)/new_nx, np.arange(new_ny)/new_ny

    return f(new_x, new_y)

def xda_changedatasize(orig_xda, new_data):
    """
    Clone an xarray DataArray but assign data of a new size.
    """
    return xr.DataArray(
        data=new_data,
        dims=orig_xda.dims,
        coords=orig_xda.coords,
        attrs=orig_xda.attrs
    )

from mpl_toolkits.axes_grid1 import make_axes_locatable

# for creating appropriately-sized colorbars for subplots
def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)

def categorical_colormap(ncolors):
    # ncolors = 1000
    colorlist = np.array(plt.cm.tab20b(np.arange(ncolors)/ncolors))
    np.random.default_rng().shuffle(colorlist)
    colorlist[0,:] = np.array([0,0,0,1])
    cm = LinearSegmentedColormap.from_list("shuffled", colorlist, N=ncolors)

    return cm

def tdiff_to_min(tdiff):
    if type(tdiff) != np.ndarray:
        tdiff = np.array(tdiff)

    return datetime_to_mins(tdiff)

def datetime_to_mins(dtm):
    return dtm.astype("timedelta64[s]").astype("float")/60

def delta_to_minutes(dt):
    return dt/(60*np.timedelta64(1, 's'))

def bits(f):
    bytes = (ord(chr(b)) for b in f)
    for b in bytes:
        for i in range(8):
            yield (b >> i) & 1

# def val_err(vals, errs=0):
#     if (type(vals) == list) or (type(vals) == np.ndarray):
#         if (type(errs) == list) or (type(errs) == np.ndarray):
#             return np.array([Complex(val, err) for val, err in zip(vals, errs)])
#         else:
#             return np.array([Complex(val, errs) for val in vals])
#     else:
#         return Complex(vals, errs)

def PlatformPath(filepath):
    if sys.platform == "linux":
        filepath = filepath.replace("\\\\", "/").replace("\\", "/")

    return Path(filepath)

# 89219-272
