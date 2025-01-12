# Spatially heterogeneous structure-function coupling in haemodynamic and electromagnetic brain networks
This repository contains processing scripts and data in support of the paper:

Liu, Z.-Q., Shafiei, G., Baillet, S., & Misic, B. (2023). Spatially heterogeneous structure-function coupling in haemodynamic and electromagnetic brain networks. _NeuroImage_, _278_, 120276. [https://doi.org/10.1016/j.neuroimage.2023.120276](https://doi.org/10.1016/j.neuroimage.2023.120276)

## `code`
The [`code`](code/) folder contains all the code used to run the analyses and generate the figures.
- [`code/script_coupling.py`](code/script_coupling.py) contains the script to reproduce Figure 1 & 2.
- [`code/script_correlation.py`](code/script_correlation.py) contains the script to reproduce Figure 3 & 4.

All code was written in Python and the required packages can be found in `requirements.txt`.

## `data`
The [`data`](data/) folder contains the data used to run the analyses. The structural and functional brain networks are derived from [HCP-YA](https://www.humanconnectome.org/study/hcp-young-adult). All data were parcellated using Schaefer400x7 atlas.
- `subj_list.txt` contains the list of 33 subjects used in the analysis.
- `sc_cons_400_nosubc.npy`, `sc_avggm_400_nosubc.npy` contains the group consensus structural connectivity matrix calculated [using a distance-dependent method](https://netneurotools.readthedocs.io/en/latest/generated/netneurotools.networks.struct_consensus.html).
- `fc_cons_400.npy` contains group average rsfMRI functional connectivity matrix
- `dist_400.npy` contains Euclidean distance matrix
- `megconn_avg_aec-lcmv_*.npy` contains group average MEG band-resolved functional connectivity matrices. For preprocessing details, check the original publications from [shafiei_megfmrimapping](https://github.com/netneurolab/shafiei_megfmrimapping) and [shafiei_megdynamics](https://github.com/netneurolab/shafiei_megdynamics).
- `Schaefer2018_400Parcels_7Networks.npy` contains resting-state network labels.
- `archemap_axis_final_wh.npy` contains cortical hierarchy annotation derived from [The Archetypal Sensorimotor-Association Axis](https://github.com/PennLINC/S-A_ArchetypalAxis).
- `bbw_intensity_profiles_400.npy` contains bigbrain intensity profiles derived from [BigBrainWarp](https://github.com/caseypaquola/BigBrainWarp).
- `sc_cplg_rsq_local.npy` contains local multivariate coupling derived from `code/script_coupling.py`.

If you use any of the HCP data, please follow their [data terms](https://www.humanconnectome.org/study/hcp-young-adult/data-use-terms), register with ConnectomeDB, and cite relevant publications. The data provided in this repository were generated using of a number of excellent open-source tools & resources, please also consider citing the relevant publications: [netneurotools](https://github.com/netneurolab/netneurotools), [neuromaps](https://github.com/netneurolab/neuromaps), [BigBrainWarp](https://github.com/caseypaquola/BigBrainWarp), [Brainstorm](https://neuroimage.usc.edu/brainstorm), [S-A_ArchetypalAxis](https://github.com/PennLINC/S-A_ArchetypalAxis), [Schaefer atlas](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal).

## `figures`
The [`figures`](figures/) folder contains the generated raw figures.
