__all__ = [
    "isnotebook",
    "ProgressParallel",
    "list_files",
    "contains_any_targets",
    "dirs_fns_to_process",
    "h5_to_dict",
    "print_h5",
    "filewrite_mode",
    "within_percentile",
    "_zoom_image",
    "_intensity_correction_mask_ndarray",
    "intensity_correction_mask",
    "resample_image",
    "xda_changedatasize",
    "colorbar",
    "categorical_colormap",
    "tdiff_to_min",
    "datetime_to_mins",
    "delta_to_minutes",
    "bits",
    "PlatformPath",
    "reorder_dict"
]

import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import os
from pathlib import Path, PurePath
import sys

from typing import Union

import xarray as xr

import joblib
import matplotlib.pyplot as plt
import numpy as np

from skimage.filters import gaussian
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

class _ProgressParallel(joblib.Parallel):
    def __call__(self, *args, **kwargs):
        with tqdm(total=_TOTAL) as self._pbar:
            return joblib.Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()

def ProgressParallel(_TOTAL, *args, **kwargs):
    return _ProgressParallel(*args, **kwargs)

def list_files(folder=".", pattern=None, exclude_in_names=["calibration",]):
    """
    List all files and folders that match `pattern`, excluding specified names
    """
    for root, folders, files in os.walk(folder):
        for filename in folders + files:
            path = PurePath(os.path.join(root, filename))
            if (pattern is None) or (path.match(pattern)):
                if not contains_any_targets(str(path), exclude_in_names):
                    yield path

def contains_any_targets(test, targets):
    return np.any([(target in test) for target in targets])

def dirs_fns_to_process(folder=".", pattern=None):
    sdt_file_list = np.array(list(list_files(folder, pattern)))
    sdt_file_dirs = np.array([fn.parent for fn in sdt_file_list])
    sdt_file_dirs_use, invs = np.unique([fn for fn in sdt_file_dirs], return_inverse=True)
    sdt_file_lists = [sdt_file_list[invs == iunq] for iunq in range(len(sdt_file_dirs_use))]

    return sdt_file_dirs_use, sdt_file_lists

def h5_to_dict(path):
    with h5py.File(path, "r") as hf:
        return { key: np.array(hf[key]) for key in list(hf.keys()) }

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

    Example usage:
    correction_mask = pc.xda_changedatasize(
        correction_mask,
        np.array([
                pc.resample_image(correction_mask.isel(file_info=0, channel=0).values, 1024, 1024),
                pc.resample_image(correction_mask.isel(file_info=0, channel=1).values, 1024, 1024)
            ])[:,:,:,np.newaxis]
    )
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

def categorical_colormap(ncolors_or_vals=200, cmap=plt.cm.tab20b):
    """
    Create a categorical colormap where 0 is mapped to black. Particularly useful for visualizing instance segmentation results.
    """

    if isinstance(ncolors_or_vals, int):
        ncolors = ncolors_or_vals
    elif isinstance(ncolors_or_vals, list):
        ncolors_or_vals = len(np.unique(np.array(ncolors_or_vals)))
    if isinstance(ncolors_or_vals, np.ndarray):
        ncolors = len(np.unique(ncolors_or_vals))

    colorlist = np.array(cmap(np.arange(ncolors)/ncolors))
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

def PlatformPath(filepath):
    if sys.platform == "linux":
        filepath = filepath.replace("\\\\", "/").replace("\\", "/")

    return Path(filepath)

def reorder_dict(the_dict, new_order):
    """
    Reorder items in dictionary
    """
    return {k: the_dict[k] for k in new_order}