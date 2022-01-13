import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from .flim import *

def plot_timelapse(df_sel, t_start=None, t_end=None, t_start2=None, t_end2=None, f_eq_mito=None, f_eq_cyto=None):
    """

    """

    df_sel = df_sel.copy()

    plot_fn = sns.scatterplot
    err_kws = {
        "linewidth": 0,
        "elinewidth": 1,
        "capsize": 3,
        "alpha": 0.4
    }

    is_mito = df_sel["channel"]=="mitochondria"
    is_cyto = df_sel["channel"]=="cytoplasm"

    fig, ax = plt.subplots(figsize=(12,16), nrows=4, ncols=2, sharex=True)

    for param, iax, ylabel in zip(["tau1", "tau2", "f", "intensity"], [ax[0,0], ax[1,0], ax[0,1], ax[1,1]], [r"$\tau_l$/ns", r"$\tau_s$/ns", r"$f$", r"intensity (AU)"]):
        plot_fn(data=df_sel, x="elapsed time", y=param, hue="channel", ax=iax, legend=False)
        iax.set_ylabel(ylabel)

        if param != "intensity":
            iax.errorbar(
                df_sel.loc[is_mito, "elapsed time"],
                df_sel.loc[is_mito, param],
                df_sel.loc[is_mito, param+"_err"],
                **err_kws
            )
            iax.errorbar(
                df_sel.loc[is_cyto, "elapsed time"],
                df_sel.loc[is_cyto, param],
                df_sel.loc[is_cyto, param+"_err"],
                **err_kws
            )

    # for param, iax in zip(["beta", "C_NADHfree", "r_ox", "J_ox"], [ax[2,0], ax[3,0], ax[2,1], ax[3,1]]):
    for param, iax, ylabel in zip(["beta", "C_NADHfree", "r_ox", "J_ox"], [ax[2,0], ax[3,0], ax[2,1], ax[3,1]], [r"$\beta$", r"$C_{NADH,free}$", r"$r_{ox}$", r"$J_{ox}$"]):
        plot_fn(data=df_sel, x="elapsed time", y=param, hue="channel", ax=iax, legend=False)
        iax.set_ylabel(ylabel)

        if param != "intensity":
            iax.errorbar(
                df_sel.loc[is_mito, "elapsed time"],
                df_sel.loc[is_mito, param],
                df_sel.loc[is_mito, param+"_err"],
                **err_kws
            )
            iax.errorbar(
                df_sel.loc[is_cyto, "elapsed time"],
                df_sel.loc[is_cyto, param],
                df_sel.loc[is_cyto, param+"_err"],
                **err_kws
            )

    for iax in ax:
        for jax in iax:
            if t_start is not None:
                jax.axvline(x=t_start, color="gray", linestyle="--")
            if t_end is not None:
                jax.axvline(x=t_end, color="gray", linestyle="--")
            if t_start2 is not None:
                jax.axvline(x=t_start2, color="gray", linestyle="--")
            if t_end2 is not None:
                jax.axvline(x=t_end2, color="gray", linestyle="--")
    ax[-1,0].set_xlabel("$t$/min")
    ax[-1,1].set_xlabel("$t$/min")

    plt.tight_layout()

    return fig, ax, df_sel

    #
    # if (t_start2 is not None) and (t_end2 is not None):
    #     use_for_eq = np.logical_and(
    #         df_sel["elapsed time"].values >= t_start2,
    #         df_sel["elapsed time"].values <= t_end2,
    #     )
    #
    #     if f_eq_mito is None:
    #         f_eq_mito = np.mean(df_sel.loc[use_for_eq & (df["channel"]=="mitochondria"), "f"].values)
    #     if f_eq_cyto is None:
    #         f_eq_cyto = np.mean(df_sel.loc[use_for_eq & (df["channel"]=="cytoplasm"), "f"].values)
