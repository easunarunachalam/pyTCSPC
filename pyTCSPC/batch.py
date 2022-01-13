import numpy as np
import pandas as pd
from pathlib import Path
import xarray as xr
import zarr

from .sdt import *

def open_zarr_to_xds(store_path):
    """
    Open a xarray dataset (containing data from multiple files) from a zarr store
    """

    group_names = sorted([int(group[0]) for group in list(zarr.open(store_path).groups())])
    ds_filenames = [str(Path(store_path).joinpath(str(group_name))) for group_name in group_names]
    ds = xr.open_mfdataset(ds_filenames, engine="zarr", consolidated=False, combine="nested", concat_dim="file_info", compat="equals")

    return ds.transpose("file_info", ...)
    # ds = xr.open_mfdataset(ds_filenames, engine="zarr", consolidated=False, concat_dim="file_info", compat="no_conflicts")
    # da = ds.to_array().squeeze("variable").reset_coords(["variable"], drop=True)
    # return da

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
