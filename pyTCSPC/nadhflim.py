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
