from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sstats
import seaborn as sns
from sklearn.linear_model import LinearRegression

from utils import group_by_index


# define paths and constants
data_base = Path("../data/")
figs_dir = Path("../figures")
freq_list = ["delta", "theta", "alpha", "beta", "lgamma", "hgamma"]


# define palette
pal_husl_7 = sns.color_palette("husl", 7, desat=0.6)
pal_husl_7_nodesat = sns.color_palette("husl", 7)


# load data
sc_avggm = np.load(data_base / "sc_avggm_400_nosubc.npy")
sc_avggm_neglog = -1 * np.log(sc_avggm / (np.max(sc_avggm) + 1))
sc_cplg_rsq_local = np.load(data_base / "sc_cplg_rsq_local.npy")
archemap_axis_final_wh = np.load(data_base / "archemap_axis_final_wh.npy")
rsn_mappings = np.load(data_base / "Schaefer2018_400Parcels_7Networks.npy")
bbw_intensity_profiles = np.load(data_base / "bbw_intensity_profiles_400.npy")


# correlation with structural weighted degree
sc_avggm_mean = np.mean(sc_avggm, axis=0)


# FIGURE 3 | correlation with structural weighted degree
fig, axes = plt.subplots(1, 7, sharex=True, sharey=True, figsize=(6.0, 2.4))
axes_flatten = axes.flatten()
for freq_it, freq_str in enumerate(["fc"] + freq_list):
    ax = axes_flatten[freq_it]
    ax.scatter(
        sc_avggm_mean,
        sc_cplg_rsq_local[freq_it, :],
        s=6,
        alpha=0.8,
        clip_on=False,
        rasterized=True,
        color=pal_husl_7[freq_it],
        label=freq_str,
    )
    reg = LinearRegression(fit_intercept=True, n_jobs=-1)
    reg.fit(sc_avggm_mean.reshape(-1, 1), sc_cplg_rsq_local[freq_it, :].reshape(-1, 1))
    plot_x = np.linspace(0, 250, 100)
    plot_y = reg.predict(plot_x.reshape(-1, 1))
    ax.plot(plot_x, plot_y, color="w", lw=4)
    ax.plot(plot_x, plot_y, color=pal_husl_7[freq_it], lw=2)
    pr, pp = sstats.pearsonr(sc_avggm_mean, sc_cplg_rsq_local[freq_it, :])
    ax.set(
        xlabel="",
        xticks=[30, 230],
        ylabel="",
        ylim=(0, 0.9),
        yticks=[0, 0.2, 0.4, 0.6, 0.8],
        title=f"{freq_str}\n{pr:.2f}",
    )
    if freq_it > 0:
        ax.tick_params(left=False)
        sns.despine(top=True, right=True, left=True, trim=True, offset=10, ax=ax)
    else:
        sns.despine(top=True, right=True, left=False, trim=True, offset=10, ax=ax)
plt.subplots_adjust(wspace=-0.3)
plt.savefig(figs_dir / "corr_sc_avggm_mean.png", dpi=300)


# FIGURE 3 | ditribution with intrinsic functional modules
fig, axes = plt.subplots(1, 7, sharex=True, sharey=True, figsize=(6.0, 2.4))
axes_flatten = axes.flatten()
for freq_it, freq_str in enumerate(["fc"] + freq_list):
    ax = axes_flatten[freq_it]
    val_by_rsn = group_by_index(sc_cplg_rsq_local[freq_it, :], rsn_mappings)
    ax.axvline(x=0.4, ls="dashed", color="lightgrey", alpha=1)
    sns.stripplot(
        data=val_by_rsn,
        ax=ax,
        dodge=True,
        color="gray",
        size=3,
        alpha=0.25,
        orient="h",
        zorder=5,
        clip_on=False,
        rasterized=True,
    )
    ax.plot(
        [np.mean(_) for _ in val_by_rsn],
        range(7),
        marker="",
        ls="-",
        color=pal_husl_7[freq_it],
        zorder=8,
    )
    ax.plot(
        [np.mean(_) for _ in val_by_rsn],
        range(7),
        marker="o",
        markersize=7,
        ls="",
        color="white",
        zorder=9,
    )
    ax.plot(
        [np.mean(_) for _ in val_by_rsn],
        range(7),
        marker="o",
        markersize=5,
        ls="",
        color=pal_husl_7_nodesat[freq_it],
        zorder=10,
    )
    ax.set(xlim=(0, 0.8), xticks=[0, 0.8], title="")
    if freq_it > 0:
        # ax.tick_params(axis="y", which="both", length=0)
        ax.tick_params(left=False)
        ax.set(yticklabels=[])
        sns.despine(top=True, right=True, left=True, trim=True, offset=10, ax=ax)
    else:
        sns.despine(top=True, right=True, left=False, trim=True, offset=10, ax=ax)
    plt.subplots_adjust(wspace=0.2)
plt.savefig(figs_dir / "corr_val_by_rsn.png", dpi=300)


# FIGURE 4 | correlation with SA axis
fig, axes = plt.subplots(1, 7, sharex=True, sharey=True, figsize=(5.8, 2.4))
axes_flatten = axes.flatten()
for freq_it, freq_str in enumerate(["fc"] + freq_list):
    ax = axes_flatten[freq_it]
    ax.scatter(
        archemap_axis_final_wh,
        sc_cplg_rsq_local[freq_it, :],
        s=6,
        alpha=0.8,
        clip_on=False,
        rasterized=True,
        color=pal_husl_7[freq_it],
        label=freq_str,
    )
    reg = LinearRegression(fit_intercept=True, n_jobs=-1)
    reg.fit(
        archemap_axis_final_wh.reshape(-1, 1),
        sc_cplg_rsq_local[freq_it, :].reshape(-1, 1),
    )
    plot_x = np.linspace(0, 400)
    plot_y = reg.predict(plot_x.reshape(-1, 1))
    ax.plot(plot_x, plot_y, color="w", lw=4)
    ax.plot(plot_x, plot_y, color=pal_husl_7[freq_it], lw=2)
    pr, pp = sstats.pearsonr(archemap_axis_final_wh, sc_cplg_rsq_local[freq_it, :])
    ax.set(
        xlabel="S-A",
        xlim=(0, 400),
        xticks=[],
        ylabel="",
        ylim=(0, 0.9),
        yticks=[0, 0.2, 0.4, 0.6, 0.8],
        title="",
    )
    if freq_it > 0:
        ax.tick_params(tick1On=False)
        sns.despine(top=True, right=True, left=True, trim=True, offset=10, ax=ax)
    else:
        sns.despine(top=True, right=True, left=False, trim=True, offset=10, ax=ax)
plt.subplots_adjust(wspace=0.3)
plt.savefig(figs_dir / "corr_archemap_axis_final_wh.png", dpi=300)


# correlation with bigbrain intensity profiles
freq_bbw_intensity_corr = np.zeros((7, bbw_intensity_profiles.shape[0]))
for freq_it, freq_str in enumerate(["fc"] + freq_list):
    for bbw_it in range(bbw_intensity_profiles.shape[0]):
        pr, ppval = sstats.pearsonr(
            sc_cplg_rsq_local[freq_it, :], -1 * bbw_intensity_profiles[bbw_it, :]
        )
        freq_bbw_intensity_corr[freq_it, bbw_it] = pr


# FIGURE 4 | correlation with bigbrain intensity profiles
fig, axes = plt.subplots(
    1, 2, gridspec_kw=dict(width_ratios=[1, 4]), figsize=(8.3, 3.5)
)
ax = axes[0]
bbw_intensity_profiles_norm = 1 - (
    bbw_intensity_profiles - np.min(bbw_intensity_profiles)
) / np.ptp(bbw_intensity_profiles)
ax.plot(
    np.mean(bbw_intensity_profiles_norm, axis=1),
    np.arange(50),
    color="tab:red",
    zorder=10,
)
for i in range(400):
    ax.plot(
        bbw_intensity_profiles_norm[:, i], np.arange(50), color="tab:gray", alpha=0.25
    )
ax.set(yticks=[], yticklabels=[], xticks=[])
ax.invert_yaxis()
ax = axes[1]
for freq_it, freq_str in enumerate(["BOLD"] + freq_list):
    ax.plot(
        freq_bbw_intensity_corr[freq_it, :],
        np.arange(50),
        color="white",
        lw=4,
        zorder=10,
    )
    ax.plot(
        freq_bbw_intensity_corr[freq_it, :],
        np.arange(50),
        color=pal_husl_7[freq_it],
        lw=3,
        label=freq_str,
        zorder=10,
    )
plt.legend(frameon=False)
ax.set(yticks=[], yticklabels=[])
ax.invert_yaxis()
sns.despine(top=True, right=True, trim=True, offset=10)
plt.savefig(figs_dir / "freq_bbw_intensity_corr.png", dpi=300)
