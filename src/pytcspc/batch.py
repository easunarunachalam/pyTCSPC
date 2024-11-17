__all__ = [
    "abbrev_coord_data",
    "open_zarr_group_to_xds",
    "zarr_groups",
    "open_zarr_to_xds",
    "concat_zarr_datasets",
    "get_intensity_images"
]

from itertools import chain
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.autonotebook import tqdm
import xarray as xr
import zarr

from .sdt import *
from .util import *

def abbrev_coord_data(coord_data, exclude_longer_than=10):
    data = coord_data.flatten()
    if len(data) == 1:
        return data[0]
    elif len(data) > exclude_longer_than:
        return []
    else:
        return data

def open_zarr_group_to_xds(store_path):
    """
    Open a xarray dataset from a single group in a zarr store
    """

    ds = xr.open_dataset(store_path, engine="zarr", consolidated=False)
    return ds #.transpose("file_info", ...)

def zarr_groups(store_path):
    """
    List group names in zarr store
    """
    # for group in list(zarr.open(store_path).groups()):
    #     print(group[0])
    group_names = sorted([int(group[0]) for group in list(zarr.open(store_path).groups())])
    return [str(Path(store_path).joinpath(str(group_name))) for group_name in group_names]

def open_zarr_to_xds(store_path, compat="equals", drop_var_names=[]):
    """
    Open a xarray dataset (containing data from multiple files) from a zarr store
    """

    # group_names = sorted([int(group[0]) for group in list(zarr.open(store_path).groups())])
    # ds_filenames = [str(Path(store_path).joinpath(str(group_name))) for group_name in group_names]
    ds_filenames = zarr_groups(store_path)

    def drop_other_vars(ds):
        for ivar in drop_var_names:
            if ivar in ds.variables:
                ds = ds.drop_vars([ivar])

        return ds

    ds = xr.open_mfdataset(ds_filenames,
        engine="zarr",
        consolidated=False,
        combine="nested",
        concat_dim="file_info",
        compat=compat,
        overwrite_encoded_chunks=True,
        preprocess=drop_other_vars
    )

    return ds.transpose("file_info", ...)
    # ds = xr.open_mfdataset(ds_filenames, engine="zarr", consolidated=False, concat_dim="file_info", compat="no_conflicts")
    # da = ds.to_array().squeeze("variable").reset_coords(["variable"], drop=True)
    # return da

def concat_zarr_datasets(zarr_stores, drop_var_names=[]):
    ds_list = []
    for store_path in tqdm(zarr_stores, position=0, desc=None):
        store_path_unstructured = Path(store_path)

        i_ds = open_zarr_to_xds(store_path_unstructured, drop_var_names=drop_var_names)
        ds_list.append(i_ds)

    ds = xr.concat(ds_list, dim="file_info")

    return ds

def reshape_for_pos_tpt(da, num_positions, ragged="trim"):
    """
    Reorganizes a xarray DataArray with a simple linear ordering of images along position and timepoint information.

    da = original DataArray

    num_positions = number of positions

    if ragged = 'trim', remove images from timepoints where a full set of positions has not been imaged
    if ragged = 'expand', pad array by filling un-imaged positions in the last time point with zeros/nans

    TODO:
    - add support for multiple Z positions
    - transfer all coords programmatically, hard-coded the important ones (channel and microtime) for now
    """

    if ragged == "trim":
        new_numel = (len(da)//num_positions)*num_positions
        newsize_da = da.head(file_info=new_numel)
        # print(newsize_da.shape)

    data = newsize_da.data
    orig_fns = np.array(newsize_da["filename"].data)
    pos_tpts = np.array([
        list(
            acq_index_to_pos_t(
                bh_acquisition_index(orig_fn),
                num_positions=num_positions
            )
        ) for orig_fn in orig_fns
    ])
    file_info_newdims = (len(data)//num_positions, num_positions,)
    data = data.reshape(file_info_newdims + data.shape[1:])

    file_info_coords = ["filename", "parent_directory", "acqtime", "numscans", "laser_period"]
    file_info_coords = [fic for fic in file_info_coords if ((fic in da.coords) and (len(newsize_da[fic].data) == new_numel))]

    new_da = xr.DataArray(
        data=data,
        dims=("timepoint", "position",) + da.dims[1:],
        coords=dict([
            (file_info_coord, (["timepoint", "position"], np.array(newsize_da[file_info_coord].data).reshape(file_info_newdims))) for file_info_coord in file_info_coords
        ])
    )

    additional_coords = dict([
        (coord, da.coords.get(coord).data) for coord in ["channel", "microtime_ns"] if coord in da.coords
    ])
    new_da = new_da.assign_coords(additional_coords)

    return new_da

def get_intensity_images(use_dir_names, zarr_location="processed_data.zarr"):
    """
    Retrieve intensity images from zarr store whose source files are within directories contained in the list `use_dir_names`
    """

    zarr_stores = list(list_files(".", pattern=zarr_location))
    zarr_groups_flat = np.array(list(chain(*[zarr_groups(str(zarr_store)) for zarr_store in tqdm(zarr_stores, desc="flatten zarr groups")])))

    parent_dirs = [open_zarr_group_to_xds(izgf).parent_directory.data for izgf in tqdm(zarr_groups_flat, desc="get parent directories")]
    parent_dirs = np.squeeze(np.array(parent_dirs))
    idx_use_parent_dirs = np.where([(str(parent_dir) in use_dir_names) for parent_dir in parent_dirs])[0]

    ims = []
    for i in tqdm(idx_use_parent_dirs, desc="retrieve intensity images"):
        i_da = open_zarr_group_to_xds( zarr_groups_flat[i] )["intensity"]
        ims.append(i_da)

    da = xr.concat(ims, dim="file_info").transpose("file_info", "y", "x")
    # da = da.isel(file_info=np.argsort(da["acqtime"].values))

    return da, parent_dirs[idx_use_parent_dirs], zarr_groups_flat[idx_use_parent_dirs]