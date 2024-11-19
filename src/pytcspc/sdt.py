__all__ = [
    "load_sdt",
    "get_acqtime",
    "sum_images",
    "intensity_image",
    "correct_intensity",
    "mean_lifetime_image",
    "calculate_mean_lifetime_images",
    "decay_curve",
    "construct_multichannel_flim_image",
    "sdt_conversion_list",
    "convert_sdts",
    "batch_convert_flim_images",
    "bh_acquisition_index",
    "acq_index_to_pos_t",
]

from datetime import datetime
import glob
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union
import numpy as np
import re
from skimage.filters import gaussian
import xarray as xr
import zarr

from sdtfile import SdtFile as raw_sdtfile

from .util import *

from tqdm.autonotebook import tqdm

def load_sdt(f, dims="CYXM", channel_names=None, dtype=np.uint32, use_dask=True):
    """
    Read becker & hickl sdt file and return FLIM image data in the form of an xarray.DataArray

    use_dask=False is recommended
    """

    if not (isinstance(f, str) or isinstance(f,Path)):
        raise TypeError("Invalid input type")

    fpath = Path(f)

    file = raw_sdtfile(str(fpath))

    acqtime = get_acqtime(file.info)

    file.data = np.array(file.data, dtype=dtype)

    times    = np.array(file.times)[0] * 1e9

    mi = file.measure_info[0]
    numscans = np.maximum(1, mi.MeasHISTInfo.fida_points[0])
    # print(mi["StopInfo"]["min_sync_rate"], mi["StopInfo"]["max_sync_rate"])
    laser_period = np.mean((1/mi["StopInfo"]["min_sync_rate"][0], 1/mi["StopInfo"]["max_sync_rate"][0]))*1e9

    dim_list = ["file_info"]
    for dim in dims:
        if dim == "C": dim_list.append("channel")
        elif dim == "Y": dim_list.append("y")
        elif dim == "X": dim_list.append("x")
        elif dim == "M": dim_list.append("microtime_ns")
        else: raise SyntaxError("Unrecognized dimension '{}'".format(dim))

    if channel_names is None:
        channel_names = ["M1", "M2", "M3", "M4"][:file.data.shape[0]]

    da_sdt = xr.DataArray(
        data=file.data[np.newaxis,:].astype(dtype),
        dims=dim_list,
        coords={
            "filename": ("file_info", [fpath.name]),
            "parent_directory": ("file_info", [str(fpath.parents[0])]),
            "acqtime": ("file_info", [acqtime]),
            "numscans": ("file_info", [numscans]),
            "laser_period": ("file_info", [laser_period]),
            "channel": channel_names,
            "microtime_ns": np.array(file.times)[0] * 1e9,
        },
    )

    if use_dask:
        da_sdt = da_sdt.chunk({"file_info": 1, "channel": 1, "microtime_ns": len(da_sdt["microtime_ns"].data)})

        if "Y" in dims:
            da_sdt = da_sdt.chunk({"y": len(da_sdt["y"].data)})
        if "X" in dims:
            da_sdt = da_sdt.chunk({"x": len(da_sdt["x"].data)})

    return da_sdt

def get_acqtime(sdtfile_info):
    """
    get acquisition time (wall clock time) for SDT
    """

    info_lines   = sdtfile_info.split("\n")
    hhmmss_str   = [i.split(":") for i in info_lines if i.startswith("  Time")][0][1:]
    hhmmss       = [int(j) for j in hhmmss_str]
    yyyymmdd_str = [i.split(":") for i in info_lines if i.startswith("  Date")][0][1]
    yyyymmdd     = [int(i) for i in yyyymmdd_str.split("-")]

    return datetime(*yyyymmdd, *hhmmss)

def sum_images(images, use_xarray_sum=True):
    if use_xarray_sum:
        return images.sum(dim="file_info", keep_attrs=True)
    else:
        return images.values.sum(axis=0)

def intensity_image(flim_image, use_xarray_sum=True, correct_mask=None, squeeze=True, normalize_num_scans=True):

    flim_image = flim_image.astype(np.float32)

    if use_xarray_sum:
        int_img = flim_image.sum(dim="microtime_ns", keep_attrs=True)
    else:
        int_img = flim_image.values.sum(axis=-1)

    if (correct_mask is not None) and (isinstance(correct_mask, xr.DataArray) or isinstance(correct_mask, np.ndarray)):
        int_img = correct_intensity(int_img, mask=correct_mask)

    if squeeze:
        int_img = int_img.squeeze()

    if normalize_num_scans:
        int_img = int_img / flim_image.numscans

    return int_img

def correct_intensity(
    raw_image:Union[xr.DataArray,np.ndarray],
    mask:Union[xr.DataArray,np.ndarray],
    manual_multiply=False,
):
    """
    Correct
    """

    if isinstance(raw_image, xr.DataArray) and isinstance(mask, xr.DataArray) and manual_multiply:

        assert raw_image.dims == mask.dims, ValueError("Raw image and mask have incompatible dimensions.")

        corrected_data = np.multiply(raw_image.data, mask.data)
        raw_image.data = corrected_data

        return raw_image

    else:
        # print("hello3")
        # if isinstance(raw_image, xr.DataArray):
        #     raw_image = raw_image.values
        #     print("hello4")
        # if isinstance(mask, xr.DataArray):
        #     mask = mask.values
        #     print("hello5")
        # print(raw_image.shape, mask.shape)
        # if raw_image.shape == mask.shape:
        #     return raw_image * mask
        return raw_image * mask


def mean_lifetime_image(input_flim_image, valid_pixel_mask=None, microtimes=None, sigma=15, subtract_constant_lifetime=0):
    """
    Calculate mean lifetime image for a particular set of pixels in a FLIM image. Values are an Gaussian-weighted average over the neighborhood of a given pixel, with scale set by `sigma`.

    Two images, each convolved with the smoothing kernel, are calculated:
    - a sum of photon lifetimes image
    - an intensity image

    Their quotient is the mean lifetime image. The calculation of each of these two images is restricted to valid pixels, which are specified by the 2d array `valid_pixel_mask`. In the final image, invalid pixels (those not used for the lifetime calculation) are zero to 0.

    It may be helpful to determine the appropriate scale for the colorbar by inspecting the histogram of nonzero values in the mean lifetime image, since there are likely to be a few outliers that end up influencing the min/max values used for the automatic colorbar range. For example,
    `plt.hist(mean_lifetime_image_blur[np.nonzero(mean_lifetime_image_blur)], bins=np.arange(4,6,0.1))`

    """

    if valid_pixel_mask is None:
        valid_pixel_mask = np.ones((input_flim_image.x.size, input_flim_image.y.size))

    if isinstance(input_flim_image, xr.DataArray):

        flim_image = input_flim_image.data
        x_size, y_size = input_flim_image.x.size, input_flim_image.y.size
        intensity_image = input_flim_image.sum(dim="microtime_ns").squeeze().data
        print("shape", intensity_image.shape)
        microtimes = input_flim_image.microtime_ns.data

    else:

        flim_image = input_flim_image
        x_size, y_size = input_flim_image.shape[1], input_flim_image.shape[0]
        intensity_image = input_flim_image.sum(axis=-1).squeeze()
        if microtimes is None:
            raise TypeError("'microtimes' array must be provided if 'input_flim_image' is not xarray.DataArray.")

    intensity_tile = np.repeat( intensity_image[:,:,np.newaxis], microtimes.size, axis=-1 )
    microtime_tile = np.tile( microtimes, (y_size, x_size, 1) )
    flim_image = flim_image.squeeze()

    sum_lifetime_image = np.multiply(
        flim_image,
        microtime_tile
    ).sum(axis=-1)

    sum_lifetime_image_masked = np.multiply(sum_lifetime_image, valid_pixel_mask)
    intensity_image_masked = np.multiply(intensity_image, valid_pixel_mask)

    sum_lifetime_blur = gaussian(sum_lifetime_image_masked, sigma=sigma, preserve_range=True).squeeze()
    intensity_blur = gaussian(intensity_image_masked, sigma=sigma, preserve_range=True).squeeze()
    
    with np.errstate(divide="ignore"):
        mean_lifetime_image_blur = np.divide(sum_lifetime_blur, intensity_blur)
        
    mean_lifetime_image_blur = np.multiply(mean_lifetime_image_blur, valid_pixel_mask)
    mean_lifetime_image_corrected = mean_lifetime_image_blur - subtract_constant_lifetime

    mean_lifetime_image_corrected[~valid_pixel_mask] = 0

    # return flim_image, microtime_tile, sum_lifetime_blur, intensity_blur, mean_lifetime_image_blur, sum_lifetime_image, mean_lifetime_image_blur, intensity_image_masked
    return mean_lifetime_image_corrected

def calculate_mean_lifetime_images(da_flim, da_segmentation, subtract_constant_lifetime=0):

    da_flim = da_flim.astype(np.float32).chunk(chunks={
        "timepoint": 1,
        "position": 1,
        "channel": 1,
        "y": len(da_flim.y.data),
        "x": len(da_flim.x.data),
        "microtime_ns": len(da_flim.microtime_ns.data),
    })

    return xr.apply_ufunc(
        mean_lifetime_image,
        da_flim,
        da_segmentation,
        kwargs={
            "microtimes": da_flim.microtime_ns.data,
            "subtract_constant_lifetime": subtract_constant_lifetime,
        },
        input_core_dims=[["y", "x", "microtime_ns"], ["y", "x"]],
        output_core_dims=[["y", "x"]],
        output_dtypes=np.float32,
        vectorize=True,
        dask="parallelized",
        # dask_gufunc_kwargs={"allow_rechunk": True},
    )

def decay_curve(
        flim_image, mask=None, normalize=False,
        fig=None, ax=None, plot=False, label=None,
        trunc=False, peak_start=2.5, peak_end=4,
        bgsub=False, bg_start=9, bg_end=11
    ):
    """
    return histogram of photon lifetimes

    Parameters
    ----------
    mask:
        if None, use decay curves for all pixels
        if a binary nxm np.ndarray, where n,m are the x,y dimensions of the image, use the decay curves for elements that are True
    nbins: number of time bins to use for decay curve. Original number of bins should be a mutiple of this number.
    normalize: if True, divide all counts by the total number of counts in all bins
    plot: if True, plot the decay curve

    """

    if isinstance(flim_image, xr.DataArray):
        if (flim_image.channel.values.size != 1):
            raise ValueError("Channel (e.g. M1, M2, NADH, Venus) not specified. Use '.sel(channel=<channel_name>)' to select detector.")

        selected_decays = flim_image.data
    else:
        selected_decays = flim_image

    n_time_bins = selected_decays.shape[-1]
    selected_decays = selected_decays.reshape((-1, n_time_bins))

    if mask is None:
        npx = 1
    else:
        if isinstance(mask, xr.DataArray):
            mask = mask.data
        mask = np.reshape(mask, (-1))
        npx = np.sum(mask)

        selected_decays = selected_decays[mask,:]

    # combine decays for all selected pixels
    # (sum values from all decays in each time bin)
    dc = selected_decays.reshape(-1, n_time_bins).sum(axis=0).astype(float)

    if bgsub:
        # is_background = (self.times > 7) & (self.times < 9)
        is_background = (flim_image["microtime_ns"].values > bg_start) & (flim_image["microtime_ns"].values < bg_end)
        background_time = flim_image["microtime_ns"].values[is_background]
        background = dc[is_background]
        mean_background = np.mean(background)
        dc -= mean_background
        dc[dc < 0] = 0

    if trunc:
        # is_peak_region = (self.times >= 1.5) & (self.times <= 3)
        is_peak_region = (flim_image["microtime_ns"].values >= peak_start) & (flim_image["microtime_ns"].values <= peak_end)
        dc[np.logical_not(is_peak_region)] = 0

    if normalize:
        dt = flim_image["microtime_ns"].values[1] - flim_image["microtime_ns"].values[0]
        dc = np.divide(dc, np.sum(dc)*dt)

    if plot:
        if (fig is None) and (ax is None):
            fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(flim_image["microtime_ns"].values, dc, label=label)
        ax.set_xlabel(r"$t$/ns")
        if normalize:
            ax.set_ylabel(r"frequency")
        else:
            ax.set_ylabel(r"photons")
        ax.set_yscale("log")
        plt.tight_layout()

        return fig, ax, dc
    else:
        return dc

def construct_multichannel_flim_image(flim_image_list, channel_names_list):
    """
    Construct a FLIM image with multiple channels from one or more multi-channel FLIM images.

    flim_image_list = list of FLIM images
    channel_names_list = list of lists, with members of upper level list matched with members of flim_image_list, and lower-level lists containing the names of the channels corresponding to each channel in

    Examples:

    da = construct_multichannel_flim_image([sdt, sdt2], [["mito", "NADH"], ["Venus", "None"]])
    """

    assert len(flim_image_list) == len(channel_names_list)

    new_flim_image_list = []
    for flim_image, channel_names in zip(flim_image_list, channel_names_list):
        if len(channel_names) > 0:
            new_flim_image = flim_image.assign_coords(channel=channel_names)
        new_flim_image = new_flim_image.drop_sel(channel="None", errors="ignore")
        new_flim_image_list.append(new_flim_image)

    if len(flim_image_list) == 1:
        return new_flim_image
    elif len(flim_image_list) > 1:
        multichannel_flim_image = xr.concat(new_flim_image_list, dim="channel", coords="all", compat="no_conflicts")
        return multichannel_flim_image
    else:
        return None

def sdt_conversion_list(
    n_acqs_per_img,
    main_dir=None,
    fns=None,
    exclude_in_names=["calibration"],
):
    """
    main_dir: parent directory containing all child directories that contain sdt file_attrs
    fns: list of sdt filenames containing absolute or relative paths (but relative paths must be unique)
    must specify either main or fns
    """

    if (main_dir is None) and (fns is not None):
        _ = _
    elif (main_dir is not None) and (fns is None):
        main_dir = Path(main_dir)
        fns = np.sort(list(list_files(folder=main_dir, pattern="*.sdt", exclude_in_names=exclude_in_names)))
    else:
        raise TypeError("Must specify either 'main_dir' or 'fns' but not both.")

    if not isinstance(exclude_in_names, list):
        exclude_in_names = [exclude_in_names]

    parent_dirs, rev_map = np.unique([fn.parent for fn in fns], return_inverse=True)
    fns_contained_list = [fns[(rev_map==i)] for i in range(len(parent_dirs))]

    multichannel_flim_image_sdt_fns = []
    for parent_dir, fns_contained in zip(parent_dirs, fns_contained_list):
        for i_img in np.arange(0, len(fns_contained), n_acqs_per_img):

            flim_img_fn_list = []

            if (i_img + n_acqs_per_img - 1) >= len(fns_contained):
                print("Not enough channels for image requiring this acquisition file: ", fns_contained[i_img])
                continue

            for i_acq_per_img in range(n_acqs_per_img):
                flim_img_fn_list.append(fns_contained[i_img + i_acq_per_img])

            multichannel_flim_image_sdt_fns.append(flim_img_fn_list)

    return multichannel_flim_image_sdt_fns

def convert_sdts(
    conversion_list,
    channel_names,
    correction_mask=None,
    type="zarr",
    archive_path="processed_data.zarr",
    sync_path="zarr_writer.sync",
    overwrite=False,
    n_jobs=4
):

    """
    Convert groups of sdt files in 'conversion_list' to multichannel FLIM images and intensity images saved in zarr format.

    Note that 'consolidated=False' when writing the zarr seems to avoid .zmetadata PermissionErrors. Change to True or None at your own risk.

    """

    for sdt_list in conversion_list:
        assert len(sdt_list) == len(channel_names), ValueError("len(channel_names) must equal number of acqisitions per image.")

    synchronizer = zarr.ProcessSynchronizer(sync_path)

    def convert_single_group(i_group):

        group_path = Path(archive_path).joinpath(f"/{int(i_group):d}")
        group_name = str(group_path)

        if group_path.exists() and (not overwrite):
            return

        sdt_fn_set = conversion_list[i_group]
        flim_img_list = [ load_sdt(Path(sdt_fn), dtype=np.uint16) for sdt_fn in sdt_fn_set]
        flim_img = construct_multichannel_flim_image(flim_img_list, channel_names)
        intensity_img = intensity_image(flim_img, correct_mask=correction_mask)

        ds = xr.Dataset(dict(
            FLIM=flim_img,
            intensity=intensity_img
        ))

        if type == "zarr":
            ds.to_zarr(archive_path, mode="a", group=group_name, synchronizer=synchronizer, consolidated=False)
        elif type == "nc":
            ds.to_zarr(archive_path, mode="a", group=group_name, synchronizer=synchronizer, consolidated=False)
        else:
            raise ValueError("Unrecognized type for saving.")

    class ProgressParallel(Parallel):
        def __call__(self, *args, **kwargs):
            with tqdm(total=len(conversion_list)) as self._pbar:
                return Parallel.__call__(self, *args, **kwargs)

        def print_progress(self):
            self._pbar.n = self.n_completed_tasks
            self._pbar.refresh()

    ProgressParallel(n_jobs=n_jobs)(delayed(convert_single_group)(i_group) for i_group in range(len(conversion_list)))

def batch_convert_flim_images(
    main_dir=None,
    fns=None,
    exclude_in_names=["calibration"],
    n_acqs_per_img=1,
    channel_names=[[]],
    correction_mask=None,
    archive_name="processed_data.zarr",
    group="unstructured",
    dry_run=False,
    test_first_only=False,
    overwrite=False,
    n_jobs=4
):
    """
    Convert sdt files in a directory to multichannel FLIM images and intensity images saved in zarr format.

    main_dir: parent directory containing all child directories that contain sdt file_attrs
    fns: list of sdt filenames containing absolute or relative paths (but relative paths must be unique)
    must specify either main or fns

    dry_run: if True, simply list the directories where conversion would be performed without loading or converting any files

    Note that 'consolidated=False' when writing the zarr seems to avoid .zmetadata PermissionErrors. Change to True or None at your own risk.

    """

    assert n_acqs_per_img == len(channel_names), ValueError("len(channel_names) must equal 'n_acqs_per_img'.")

    if (main_dir is None) and (fns is not None):
        _
    elif (main_dir is not None) and (fns is None):
        main_dir = Path(main_dir)
        fns = np.sort([Path(fp) for fp in glob.glob(str(main_dir.joinpath("**/*.sdt")), recursive=True) if not contains_any_targets(str(fp), exclude_in_names)])
    else:
        raise TypeError("Must specify either 'main_dir' or 'fns' but not both.")

    if not isinstance(exclude_in_names, list):
        exclude_in_names = [exclude_in_names]

    parent_dirs, rev_map = np.unique([fn.parent for fn in fns], return_inverse=True)
    fns_contained_list = [fns[(rev_map==i)] for i in range(len(parent_dirs))]

    for parent_dir, fns_contained in zip(parent_dirs, fns_contained_list):
        archive_path = parent_dir.joinpath(archive_name)

        if dry_run or test_first_only:
            print(archive_path)
            n_jobs = 1

        synchronizer = zarr.ProcessSynchronizer(parent_dir.joinpath("zarr_writer.sync"))

        if archive_path.exists():
            existing_groups = [i_group[0] for i_group in list(zarr.open(str(archive_path), mode="r").groups())]
        else:
            existing_groups = []

        if (not archive_path.exists()) or overwrite:
            initial_mode = "w"
        else:
            initial_mode = "a"


        def process_single_image_set(i_img, i_group):

            if (not overwrite) and (str(i_group) in existing_groups):
                return

            flim_img_list = []

            if (i_img + n_acqs_per_img - 1) >= len(fns_contained):
                print("Not enough channels for image requiring this acquisition file: ", fns_contained[i_img])
                return

            if dry_run:
                print(end="   ")

            for i_acq_per_img in range(n_acqs_per_img):

                if dry_run:
                    print(fns_contained[i_img + i_acq_per_img], end=" ")
                else:
                    sdt = load_sdt(fns_contained[i_img + i_acq_per_img], dtype=np.uint16)
                    flim_img_list.append(sdt)

            if dry_run:
                print("")
            else:
                flim_img = construct_multichannel_flim_image(flim_img_list, channel_names)
                intensity_img = intensity_image(flim_img, correct_mask=correction_mask)
                # return flim_img, intensity_img
                ds = xr.Dataset(dict(
                    FLIM=flim_img,
                    intensity=intensity_img
                ))

                if test_first_only:
                    return ds

                if i_img == 0:
                    ds.to_zarr(archive_path, mode="w", group=group + f"/{int(i_group):d}", synchronizer=synchronizer, consolidated=False)
                else:
                    ds.to_zarr(archive_path, mode="a", group=group + f"/{int(i_group):d}", synchronizer=synchronizer, consolidated=False)

                del sdt
                del flim_img
                del intensity_img
                del ds

        n_groups = np.ceil(len(fns_contained)/n_acqs_per_img)
        processing_iterator = list(zip(np.arange(0, len(fns_contained), n_acqs_per_img), np.arange(n_groups)))
        if not dry_run:
            processing_iterator = tqdm(processing_iterator, total=n_groups, desc=str(parent_dir))
        # , tqdm_args=dict(desc=str(parent_dir))
        if n_jobs > 1:
            # _TOTAL = len(processing_iterator)
            Parallel(n_jobs=n_jobs)(
                    delayed(process_single_image_set)(i_img, i_group) for i_img, i_group in processing_iterator
            )
        else:
            for i_img, i_group in processing_iterator:
                if test_first_only:
                    return process_single_image_set(i_img, i_group)
                else:
                    process_single_image_set(i_img, i_group)


def bh_acquisition_index(orig_filename):
    """
    Given the name of a file that was automatically named by bh SPCM software, convert to acquisition number.

    This can be useful for determining acquisition position or other properties from the filename.
    """

    orig_filename = str(orig_filename)

    return int(re.search(r"_c(\d+)\.", orig_filename).group(0)[2:-1])

def acq_index_to_pos_t(acq_idx, num_positions=1):
    return (acq_idx-1) % num_positions, (acq_idx-1) // num_positions
