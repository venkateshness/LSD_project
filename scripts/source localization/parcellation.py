
#%%
import numpy as np
import mne
from nilearn import plotting
from nilearn.regions import signals_to_img_labels
from nilearn.datasets import fetch_icbm152_2009

HOMEDIR = "/users/local/Venkatesh/LSD_project"
path_Glasser = f"{HOMEDIR}/src_data/Glasser_masker.nii.gz"
mnitemp = fetch_icbm152_2009()

with np.load(
    f"{HOMEDIR}/src_data/sourcespace_to_glasser_labels.npz"
) as dobj:  # shoutout to https://github.com/rcruces/2020_NMA_surface-plot.git
    atlas = dict(**dobj)


def averaging_by_parcellation(sub):
    """Aggregating the native brain surface fsaverage vertices to parcels (180 each hemi)

    Args:
        sub (array): source time course per subject

    Returns:
        source_signal_in_parcels : signal in parcels
    """
    source_signal_in_parcels = list()
    
    for roi in list(set(atlas["labels_R"]))[:-1]:
            source_signal_in_parcels.append(
                np.mean(sub[10242:][np.where(roi == atlas["labels_R"])], axis=0)
            )
    
    for roi in list(set(atlas["labels_L"]))[:-1]:
            source_signal_in_parcels.append(
                np.mean(sub[:10242][np.where(roi == atlas["labels_L"])], axis=0)
            )

    return source_signal_in_parcels

#%%
morphedstc = np.load('/users/local/Venkatesh/LSD_project/src_data/derivatives/func/Music/LSD/sub-003/meg/source_estimates/sub-003_source_estimate.npz', allow_pickle=True)['arr_0']

morphedstc_parcellated = averaging_by_parcellation(morphedstc[0])
# %%

data_to_plot_morphed = np.mean(morphedstc_parcellated, axis=(1))
zscore = (data_to_plot_morphed - np.mean(data_to_plot_morphed))/np.std(data_to_plot_morphed)
nifti= signals_to_img_labels(zscore, path_Glasser, mnitemp["mask"])

plotting.plot_img_on_surf(stat_map=nifti, views=["lateral", "medial"], hemispheres=["left", "right"], symmetric_cbar=True, threshold=np.percentile(data_to_plot_morphed, 25))
# plotting.plot_glass_brain(nifti, title='Glasser Parcellation', colorbar=True, threshold=None)

# %%
stc = np.load('/users/local/Venkatesh/LSD_project/src_data/derivatives/func/Music/LSD/sub-005/meg/source_estimates/sub-005_source_estimate.npz', allow_pickle=True)['stc']
#%%
stc_parcellated_all = []
for i in range(len(stc)):
    stc_parcellated = averaging_by_parcellation(stc[i])
    stc_parcellated_all.append(stc_parcellated)

#%%
stc_parcellated_all = np.array(stc_parcellated_all)
data_to_plot = np.mean(stc_parcellated_all, axis=(0,2))
zscore = (data_to_plot - np.mean(data_to_plot))/np.std(data_to_plot)
nifti= signals_to_img_labels(zscore, path_Glasser, mnitemp["mask"])

plotting.plot_img_on_surf(stat_map=nifti, views=["lateral", "medial"], hemispheres=["left", "right"], symmetric_cbar=True)
# plotting.plot_glass_brain(nifti, title='Glasser Parcellation', colorbar=True, threshold=None)
#%%
np.shape(stc_parcellated_all)
# %%
np.isclose(stc[1].data, stc[0].data, rtol=1e-20, atol=1e-20)
# %%

stc = np.load('/users/local/Venkatesh/LSD_project/src_data/derivatives/func/Music/LSD/sub-005/meg/source_estimates/sub-003_source_estimate.npz', allow_pickle=True)['arr_0']

# %%

sub = '010'

stc_parcellated = np.load(f'/users/local/Venkatesh/LSD_project/src_data/derivatives/func/Music/PLA/sub-{sub}/meg/source_estimates/sub-{sub}_source_estimate_parcellated.npz')['stc_parcellated']


# %%
import scipy
psd_all = []
for i in range(len(stc_parcellated)):
    psd_region = []
    for region in range(360):
        
        f, psd = scipy.signal.welch(stc_parcellated[i][region], fs=500, nperseg=256)
        psd_region.append(psd)
        
    psd_all.append(psd_region)


import matplotlib.pyplot as plt

# %%
# %%
from scipy.stats import sem

psd_time = np.mean(psd_all, axis=0)
psd_all_mean = np.mean(psd_time, axis=0)
psd_all_sem = np.std(psd_time, axis=0)  # Standard error of the mean for CI

# Plotting the mean PSD with confidence interval
plt.plot(f, psd_all_mean, label="Mean PSD")
plt.fill_between(f, psd_all_mean - psd_all_sem, psd_all_mean + psd_all_sem, color='blue', alpha=0.3, label="95% CI")

# Adding vertical spans for specified frequency bands
# plt.axvspan(8, 12, color='red', alpha=0.2, label="")

# Adding labels and legend
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD')
plt.legend()
plt.show()
# %%

stc_parcellated = np.array(stc_parcellated)
data_to_plot = np.mean(stc_parcellated, axis=(0,2))
zscore = (data_to_plot - np.mean(data_to_plot))/np.std(data_to_plot)
nifti= signals_to_img_labels(zscore, path_Glasser, mnitemp["mask"])

plotting.plot_img_on_surf(stat_map=nifti, views=["lateral", "medial"], hemispheres=["left", "right"], symmetric_cbar=True)
# %%
