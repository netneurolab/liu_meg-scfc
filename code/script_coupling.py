from pathlib import Path

import brainconn as bc
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sstats
import seaborn as sns
from netneurotools.metrics import communicability_wei
from netneurotools.stats import get_dominance_stats
from sklearn.linear_model import LinearRegression

from utils import navigation_wu, search_information


# define paths and constants
data_base = Path("../data/")
figs_dir = Path("../figures")
freq_list = ["delta", "theta", "alpha", "beta", "lgamma", "hgamma"]
mat_dim = 400
iu = np.triu_indices(mat_dim, 1)


# define palette
pal_husl_7 = sns.color_palette("husl", 7, desat=0.6)


# load data
fc_cons = np.load(data_base / "fc_cons_400.npy")
dist_mat = np.load(data_base / "dist_400.npy")
sc_avggm = np.load(data_base / "sc_avggm_400_nosubc.npy")
sc_avggm_neglog = -1 * np.log(sc_avggm / (np.max(sc_avggm) + 1))


# derive communication measures
spl_mat, sph_mat, _ = bc.distance.distance_wei_floyd(sc_avggm_neglog)
nsr, nsr_n, npl_mat_asym, nph_mat_asym, nav_paths = navigation_wu(dist_mat, sc_avggm)
npe_mat_asym = 1 / npl_mat_asym
npe_mat = (npe_mat_asym + npe_mat_asym.T) / 2
sri_mat_asym = search_information(sc_avggm, sc_avggm_neglog)
sri_mat = (sri_mat_asym + sri_mat_asym.T) / 2
cmc_mat = communicability_wei(sc_avggm)
mfpt_mat_asym = bc.distance.mean_first_passage_time(sc_avggm)
dfe_mat_asym = 1 / mfpt_mat_asym
dfe_mat = (dfe_mat_asym + dfe_mat_asym.T) / 2

x_comm_mats = [dist_mat, spl_mat, npe_mat, sri_mat, cmc_mat, dfe_mat]
x_comm_names = ["dist", "spl", "npe", "sri", "cmc", "dfe"]


# calculate global univariate coupling
# sc_corr_rsq_global (7,)
sc_corr_rsq_global, sc_corr_rsq_global_noadj = [], []
for freq_it, freq_str in enumerate(["fc"] + freq_list):
    reg = LinearRegression(fit_intercept=True, n_jobs=-1)
    if freq_str == "fc":
        curr_mat = fc_cons.copy()
    else:
        curr_mat = np.load(data_base / f"megconn_avg_aec-lcmv_{freq_str}.npy")
    X_zs = sstats.zscore(sc_avggm[iu], ddof=1)
    X = X_zs.reshape(-1, 1)
    y = curr_mat[iu]
    reg_res = reg.fit(X, y)
    yhat = reg.predict(X)
    SS_Residual = sum((y - yhat) ** 2)
    SS_Total = sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (float(SS_Residual)) / SS_Total
    adjusted_r_squared = 1 - (1 - r_squared) * (len(y) - 1) / (len(y) - X.shape[1] - 1)
    sc_corr_rsq_global.append(adjusted_r_squared)
    sc_corr_rsq_global_noadj.append(r_squared)


# FIGURE 2 | global univariate
fig, ax = plt.subplots(figsize=(2.5, 1.6))
ax.bar(np.arange(7), sc_corr_rsq_global, color="tab:gray", width=0.6)
ax.set(xticks=range(7), yticks=[0, 0.01, 0.02, 0.03, 0.04, 0.05])
sns.despine(top=True, right=True, trim=True, offset=5)
ax.set_xticklabels(["BOLD"] + freq_list, rotation=45, ha="right")
plt.savefig(figs_dir / "sc_corr_rsq_global.png", dpi=300)


# calculate local univariate coupling
# sc_corr_rsq_local (7, mat_dim)
sc_corr_rsq_local = []
for freq_it, freq_str in enumerate(["fc"] + freq_list):
    curr_list = []
    if freq_str == "fc":
        curr_mat = fc_cons.copy()
    else:
        curr_mat = np.load(data_base / f"megconn_avg_aec-lcmv_{freq_str}.npy")
    for i in range(mat_dim):
        reg = LinearRegression(fit_intercept=True, n_jobs=-1)
        X_no_diag = np.delete(sc_avggm[:, i], i, axis=0)
        Y_no_diag = np.delete(curr_mat[:, i], i, axis=0)
        X_zs = sstats.zscore(X_no_diag, ddof=1)
        X = X_zs.reshape(-1, 1)
        y = Y_no_diag
        reg_res = reg.fit(X, y)
        yhat = reg.predict(X)
        SS_Residual = sum((y - yhat) ** 2)
        SS_Total = sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (float(SS_Residual)) / SS_Total
        adjusted_r_squared = 1 - (1 - r_squared) * (len(y) - 1) / (
            len(y) - X.shape[1] - 1
        )
        curr_list.append(adjusted_r_squared)
    sc_corr_rsq_local.append(curr_list)
sc_corr_rsq_local = np.array(sc_corr_rsq_local)


# FIGURE 2 | local univariate
fig, ax = plt.subplots(figsize=(2.5, 1.6))
sns.stripplot(
    data=sc_corr_rsq_local.tolist(),
    size=2,
    zorder=5,
    dodge=False,
    alpha=0.8,
    palette=pal_husl_7,
    ax=ax,
    rasterized=True,
)
ax.plot(sc_corr_rsq_global, lw=4, color="white", zorder=10)
ax.plot(sc_corr_rsq_global, lw=2, color="tab:red", zorder=15)
ax.set(xticks=range(7), yticks=[0, 0.1, 0.2, 0.3])
sns.despine(top=True, right=True, trim=True, offset=5)
ax.set_xticklabels(["BOLD"] + freq_list, rotation=45, ha="right")
plt.savefig(figs_dir / "sc_corr_rsq_local.png", dpi=300)


# calculate global multivariate coupling
# sc_cplg_rsq_global (7,)
sc_cplg_rsq_global = []
for freq_it, freq_str in enumerate(["fc"] + freq_list):
    reg = LinearRegression(fit_intercept=True, n_jobs=-1)
    if freq_str == "fc":
        curr_mat = fc_cons.copy()
    else:
        curr_mat = np.load(data_base / f"megconn_avg_aec-lcmv_{freq_str}.npy")
    X_zs = sstats.zscore(
        np.c_[
            dist_mat[iu],
            spl_mat[iu],
            npe_mat[iu],
            sri_mat[iu],
            cmc_mat[iu],
            dfe_mat[iu],
        ],
        ddof=1,
    )
    X = X_zs
    y = curr_mat[iu]
    reg_res = reg.fit(X, y)
    yhat = reg.predict(X)
    SS_Residual = sum((y - yhat) ** 2)
    SS_Total = sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (float(SS_Residual)) / SS_Total
    adjusted_r_squared = 1 - (1 - r_squared) * (len(y) - 1) / (len(y) - X.shape[1] - 1)
    sc_cplg_rsq_global.append(adjusted_r_squared)


# calculate local multivariate coupling
# sc_cplg_rsq_local (7, mat_dim)
sc_cplg_rsq_local = []
for freq_it, freq_str in enumerate(["fc"] + freq_list):
    curr_list = []
    if freq_str == "fc":
        curr_mat = fc_cons.copy()
    else:
        curr_mat = np.load(data_base / f"megconn_avg_aec-lcmv_{freq_str}.npy")
    for i in range(mat_dim):
        reg = LinearRegression(fit_intercept=True, n_jobs=-1)
        X_no_diag = np.delete(
            np.c_[
                dist_mat[:, i],
                spl_mat[:, i],
                npe_mat[:, i],
                sri_mat[:, i],
                cmc_mat[:, i],
                dfe_mat[:, i],
            ],
            i,
            axis=0,
        )
        Y_no_diag = np.delete(curr_mat[:, i], i, axis=0)
        X_zs = sstats.zscore(X_no_diag, ddof=1)
        X = X_zs
        y = Y_no_diag
        reg_res = reg.fit(X, y)
        yhat = reg.predict(X)
        SS_Residual = sum((y - yhat) ** 2)
        SS_Total = sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (float(SS_Residual)) / SS_Total
        adjusted_r_squared = 1 - (1 - r_squared) * (len(y) - 1) / (
            len(y) - X.shape[1] - 1
        )
        curr_list.append(adjusted_r_squared)
    sc_cplg_rsq_local.append(curr_list)
sc_cplg_rsq_local = np.array(sc_cplg_rsq_local)
np.save(data_base / "sc_cplg_rsq_local.npy", sc_cplg_rsq_local)


# calculate dominance metrics for global multivariate coupling
dom_global_list = []
for freq_str in ["fc"] + freq_list:
    X_zs = sstats.zscore(
        np.c_[
            dist_mat[iu],
            spl_mat[iu],
            npe_mat[iu],
            sri_mat[iu],
            cmc_mat[iu],
            dfe_mat[iu],
        ],
        ddof=1,
    )
    X = X_zs
    if freq_str == "fc":
        y = fc_cons[iu]
    else:
        y = np.load(data_base / f"megconn_avg_aec-lcmv_{freq_str}.npy")[iu]
    model_metrics, model_r_sq = get_dominance_stats(X, y)
    dom_global_list.append((model_metrics, model_r_sq))

dom_global_total = [_[0]["total_dominance"] for _ in dom_global_list]
dom_global_total = np.array(dom_global_total)


# FIGURE 2 | global multivariate + global dominance
pal_set2_6 = sns.color_palette("Set2", 6, desat=0.6)
dom_global_total_ratio = dom_global_total / np.sum(
    dom_global_total, axis=1, keepdims=True
)
fig, ax = plt.subplots(figsize=(2.5, 1.6))
for comm_it, comm_name in enumerate(x_comm_names):
    if comm_it == 0:
        ax.bar(
            np.arange(7),
            dom_global_total_ratio[:, comm_it],
            width=0.6,
            label=comm_name,
            color=pal_set2_6[comm_it],
            alpha=0.8,
        )
    else:
        ax.bar(
            np.arange(7),
            dom_global_total_ratio[:, comm_it],
            width=0.6,
            bottom=np.sum(dom_global_total_ratio[:, :comm_it], axis=1),
            label=comm_name,
            color=pal_set2_6[comm_it],
            alpha=0.8,
        )
plt.legend(frameon=False)
ax.plot(sc_cplg_rsq_global, lw=5, color="white")
ax.plot(sc_cplg_rsq_global, lw=3, color="tab:red")
sns.despine(top=True, right=True, trim=True, offset=5)
ax.set(xticks=range(7))
ax.set_xticklabels(["BOLD"] + freq_list, rotation=45, ha="right")
plt.savefig(figs_dir / "dom_global_total_ratio.png", dpi=300)


# FIGURE 2 | all coupling
fig, axes = plt.subplots(7, 1, sharex=True, figsize=(4.6, 4.4))
axes_iter = iter(axes.flatten())
for freq_it, freq_str in enumerate(["fc"] + freq_list):
    ax = next(axes_iter)
    ax.hist(
        x=sc_corr_rsq_local[freq_it, :], bins=50, color=cm.tab20.colors[1], alpha=0.9
    )
    ax.hist(
        x=sc_cplg_rsq_local[freq_it, :], bins=50, color=cm.tab20.colors[0], alpha=0.5
    )
    ax.axvline(
        x=sc_corr_rsq_global[freq_it], ymin=0, ymax=50, lw=1.5, color=cm.tab20.colors[7]
    )
    ax.axvline(
        x=sc_cplg_rsq_global[freq_it], ymin=0, ymax=50, lw=1.5, color=cm.tab20.colors[6]
    )
    ax.set(
        xlim=(-0.05, 0.9),
        xticks=[0, 0.2, 0.4, 0.6, 0.8],
        ylim=(0, 25),
        yticks=[],
        ylabel="",
    )
    ax.tick_params(axis="both", which="both", length=0)
    sns.despine(top=True, right=True, left=True)
plt.savefig(figs_dir / "sc_corr-cplg_rsq_local-global.png", dpi=300)
