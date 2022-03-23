import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from .flim import *

def plot_timelapse(df_sel, t_start=None, t_end=None, t_start2=None, t_end2=None, f_eq_mito=None, f_eq_cyto=None, plot_fn=sns.scatterplot, fig=None, ax=None,):
    """

    """

    # df_sel = df_sel.copy()

    line_kws = {
        "marker": "s",
        "markerfacecolor": "w",
        "linewidth": 0,
        "alpha": 0.8
    }

    err_kws = {
        "elinewidth": 1,
        "capsize": 3,
    }

    if ax is None:
        fig, ax = plt.subplots(figsize=(9,12), nrows=4, ncols=2, sharex=True)

    for param, iax, ylabel in zip(["tau1", "tau2", "f", "intensity_norm_marker"], [ax[0,0], ax[1,0], ax[0,1], ax[1,1]], [r"$\tau_l$/ns", r"$\tau_s$/ns", r"$f$", r"intensity (AU)"]):

        for pf in np.unique(df_sel["photons_from"].values):

            if param != "intensity_norm_marker":
                iax.errorbar(
                    df_sel.loc[df_sel["photons_from"] == pf, "elapsed time"],
                    df_sel.loc[df_sel["photons_from"] == pf, param],
                    df_sel.loc[df_sel["photons_from"] == pf, param+"_err"],
                    **line_kws,
                    **err_kws
                )
            else:
                iax.plot(
                    df_sel.loc[df_sel["photons_from"] == pf, "elapsed time"],
                    df_sel.loc[df_sel["photons_from"] == pf, param],
                    **line_kws
                )
        iax.set_ylabel(ylabel)

    # for param, iax in zip(["beta", "C_NADHfree", "r_ox", "J_ox"], [ax[2,0], ax[3,0], ax[2,1], ax[3,1]]):
    for param, iax, ylabel in zip(["beta", "CNADHf", "rox", "JFLIM"], [ax[2,0], ax[3,0], ax[2,1], ax[3,1]], [r"$\beta$", r"$C_{NADH,free}$", r"$r_{ox}$", r"$J_{ox}$"]):
        # plot_fn(data=df_sel, x="elapsed time", y=param, hue="channel", ax=iax, legend=False)
        for pf in np.unique(df_sel["photons_from"].values):
            iax.set_ylabel(ylabel)
            iax.plot(
                df_sel.loc[df_sel["photons_from"] == pf, "elapsed time"],
                df_sel.loc[df_sel["photons_from"] == pf, param],
                **line_kws
            )
        iax.set_ylabel(ylabel)

    for iax in ax:
        for jax in iax:
            ylims = jax.get_ylim()
            jax.fill_betweenx(ylims, t_start, t_end, color="crimson", alpha=0.2, zorder=0)
            jax.fill_betweenx(ylims, t_start2, t_end2, color="cornflowerblue", alpha=0.2, zorder=0)
    ax[-1,0].set_xlabel("$t$/min")
    ax[-1,1].set_xlabel("$t$/min")

    plt.tight_layout()

    return fig, ax

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
