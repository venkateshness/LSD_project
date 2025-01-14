#%%

import mne
import numpy as np
from scipy.stats import ttest_rel
import scipy.stats

# Define subjects and conditions
subjects = ['003', '005', '006', '009', '010', '013', '015', '016', '017', '018']
conditions = ['LSD', 'PLA']
base_path = '/users/local/Venkatesh/LSD_project/src_data/derivatives/func/Video'
#%%
# Define canonical frequency bands
freq_bands = {
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 15),
    "Beta": (15, 30),
    "Gamma": (30, 49)
}

# Initialize storage for band power data
band_power_data = {condition: {band: [] for band in freq_bands} for condition in conditions}

# Loop through each subject and condition
for subj in subjects:
    for condition in conditions:
        file_path = f'{base_path}/{condition}/sub-{subj}/meg/sub-{subj}_cleaned_epochs_meg.fif'
        
        # Read epochs
        epochs = mne.read_epochs(file_path)
        
        # Compute PSD
        psd = epochs.compute_psd().plot()
        psd_data = psd.get_data()  
        psd_data = 10 * np.log10(psd_data)  # Convert to dB
        freqs = psd.freqs  # Array of frequency points
        
        # Compute band power by averaging within each frequency band
        for band_name, (fmin, fmax) in freq_bands.items():
            # Find indices of frequencies within the band
            band_indices = np.where((freqs > fmin) & (freqs <= fmax))[0]
            
            # Integrate PSD over the band (sum across frequencies in the band)
            band_power = psd_data[:, :, band_indices].sum(axis=-1)  # Shape: (n_epochs, n_channels)
            
            # Average across epochs for simplicity
            band_power_avg = band_power.mean(axis=0)  # Shape: (n_channels,)
            
            # Store the data
            band_power_data[condition][band_name].append(band_power_avg)

#%%
import matplotlib.pyplot as plt
from mne.stats import permutation_cluster_test

band_power_arrays = {
    condition: {band: np.array(data) for band, data in bands.items()}
    for condition, bands in band_power_data.items()
}

# Perform statistical testing for each band
stats_results = {}
for band_name in freq_bands:
    # Extract band power for each condition
    power_LSD = band_power_arrays['LSD'][band_name]  # Shape: (n_subjects, n_channels)
    power_PLA = band_power_arrays['PLA'][band_name]  # Shape: (n_subjects, n_channels)
    
    # Perform paired t-test across subjects for each channel
    t_vals, p_vals = ttest_rel(power_LSD, power_PLA, axis=0)  # Shape: (n_channels,)
    
    # print(np.shape(power_PLA))
    power_diff = power_LSD - power_PLA
    
    # adjacency,_ = mne.channels.find_ch_adjacency(epochs.info, "mag")
    
    # T_obs, cluster, cluster_pv, H0 = mne.stats.permutation_cluster_1samp_test(power_diff, n_permutations=1000, tail=0, adjacency=adjacency)
        
    
    # def shuffle_channels(data):
    #     permuted_data = np.empty_like(data)  # Placeholder for shuffled data
    #     for i in range(data.shape[0]):       # Loop over subjects
    #         np.random.seed(i)

    #         permuted_data[i] = np.random.permutation(data[i])  # Shuffle channels for each subject
    #     return permuted_data

    # t_perm = []
    # for perm in range(5000):
    #     permuted_data = shuffle_channels(power_PLA)

    #     t, _ = ttest_rel(power_LSD, permuted_data, axis=0)
        
    #     t_perm.append(t)

    # p_corrected = sum(np.abs(t_perm)>=np.abs(t_vals))/5000

    p_vals_corrected = fdrcorrection(p_vals)[0]
    t_vals = t_vals * (p_vals_corrected)
    # Store results
    stats_results[band_name] = {"t_vals": t_vals, "p_vals": p_vals}

# Plotting results for all frequency bands
fig, axes = plt.subplots(1, len(freq_bands), figsize=(20, 5), constrained_layout=True)


# Calculate the maximum absolute t-value across all bands
max_abs_t_val = max(abs(result['t_vals']).max() for result in stats_results.values())

# Define a common vmin and vmax for all plots
vmin, vmax = -max_abs_t_val, max_abs_t_val

fig, axes = plt.subplots(nrows=1, ncols=len(stats_results), figsize=(15, 3))

for ax, (band_name, result) in zip(axes, stats_results.items()):
    t_vals = result["t_vals"]
    alpha_threshold = 0.05
    significant_mask = result['p_vals'] < alpha_threshold  # Mask for significant p-values

    # Identify and drop reference magnetometer channels
    ref_channels = mne.pick_types(epochs.info, meg=False, ref_meg=True)
    ref_channel_names = [epochs.ch_names[ch] for ch in ref_channels]
    epochs.drop_channels(ref_channel_names)
    
    # Visualize t-values for this band
    im, _ = mne.viz.plot_topomap(
        t_vals, epochs.info, show=False, axes=ax, cmap='seismic',
        vlim=(vmin,vmax),  # Use the common color scale
        mask=significant_mask,
        extrapolate='head', sphere=0.09,
    )
    ax.set_title(f"{band_name} Band", fontsize=12)

# Add a colorbar for the topomaps
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal', shrink=0.6, pad=0.1)
cbar.set_label('T-values', fontsize=12)

#
# %%
stats_results['Alpha']['t_vals']

# %%
epochs.shape
# %%
import numpy as np
from scipy.signal import welch
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
conditions = ['LSD', 'PLA']

# Parameters
fs = 500  # Sampling frequency (Hz)
n_permutations = 1000  # Number of permutations
freq_bands = {
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 100),
}
subjects = ['003', '005', '009', '010', '013', '015', '016', '018']

# Placeholder for PSD and statistics
psd_data = {condition: {band: [] for band in freq_bands} for condition in conditions}

# Loop through each subject and condition
for subj in subjects:
    for condition in conditions:
        # Load parcellated source-level signal (shape: [n_epochs, n_regions, n_samples])
        file_path = f'{base_path}/{condition}/sub-{subj}/meg/source_estimates/sub-{subj}_source_estimate_parcellated.npz'
        source_data = np.load(file_path)['stc_parcellated']  # Load the parcellated data
        
        # source_data = (source_data_raw - source_data_raw.mean(axis=1, keepdims=True)) / source_data_raw.std(axis=1, keepdims=True)

        
        # Initialize storage for band power
        band_power_all_epochs = {band: [] for band in freq_bands}
        
        # Compute PSD for each epoch and region
        for epoch_data in source_data:  # Iterate over epochs (shape: [n_regions, n_samples])
            band_power_epoch = {}
            for region_idx in range(epoch_data.shape[0]):  # Iterate over regions
                region_signal = epoch_data[region_idx]  # Shape: (n_samples,)
                
                # Compute PSD using Welch's method
                f, psd = welch(region_signal, fs=fs)  # Shape: (n_freqs,)
                
                # Compute power for each frequency band
                for band_name, (fmin, fmax) in freq_bands.items():
                    band_indices = np.where((f >= fmin) & (f <= fmax))[0]
                    band_power = psd[band_indices].mean()  # Average over the band
                    band_power_epoch.setdefault(band_name, []).append(band_power)
            
            # Aggregate power across regions for this epoch
            for band_name in freq_bands:
                band_power_all_epochs[band_name].append(np.array(band_power_epoch[band_name]))

        # Average band power across epochs
        for band_name in freq_bands:
            psd_data[condition][band_name].append(np.mean(band_power_all_epochs[band_name], axis=0))


#%%
import scipy
from statsmodels.stats.multitest import fdrcorrection
psd_arrays = {
    condition: {band: np.array(data) for band, data in bands.items()}
    for condition, bands in psd_data.items()
}

# Perform statistical testing at the parcel level
stats_results = {}
for band_name in freq_bands:
    power_LSD = psd_arrays['LSD'][band_name]  # Shape: (n_subjects, n_regions)
    power_PLA = psd_arrays['PLA'][band_name]  # Shape: (n_subjects, n_regions)
    
    power_diff = power_LSD - power_PLA
    
    # T_obs, p_vals, H0 = mne.stats.permutation_t_test(power_diff, n_permutations=n_permutations, tail=0)
    t, p_vals = scipy.stats.ttest_rel(power_LSD, power_PLA, axis=0)
    # p_vals_corrected = fdrcorrection(p_vals)[1]
    
    t_vals = t * (p_vals<0.05)
    # t_vals = t * (p_vals<0.05)
    # Store results
    stats_results[band_name] = {"t_vals": t_vals}

# Visualize corrected p-values for each band
fig, axes = plt.subplots(1, len(freq_bands), figsize=(20, 5), constrained_layout=True)
parcel_labels = [f"Region {i+1}" for i in range(power_LSD.shape[1])]  # Placeholder for region labels

for ax, (band_name, result) in zip(axes, stats_results.items()):
    t_vals = result["t_vals"]
    
    
    # Plot t-values
    ax.bar(parcel_labels, t_vals, color='blue', alpha=0.7, label='T-values')
    
    # Highlight significant parcels
    ax.scatter(
        np.array(t_vals),
        t_vals,
        color='red', label='Significant (p < 0.05)'
    )
    
    ax.set_title(f"{band_name} Band", fontsize=12)
    ax.set_xlabel("Regions")
    ax.set_ylabel("T-values")
    ax.legend()

plt.show()

# %%
from nilearn import plotting
from nilearn.regions import signals_to_img_labels
from nilearn.datasets import fetch_icbm152_2009

path_Glasser = '/users/local/Venkatesh/LSD_project/src_data/Glasser_masker.nii.gz'
mnitemp = fetch_icbm152_2009()

for band in list(freq_bands.keys()):
    tvals = stats_results[band]['t_vals']
    nifti= signals_to_img_labels(tvals, path_Glasser, mnitemp["mask"])

    plotting.plot_img_on_surf(stat_map=nifti, views=["lateral", "medial"], hemispheres=["left", "right"], symmetric_cbar=True, threshold=0.000001, title=f'{band}; Negative: LSD > PLA; Positive: PLA > LSD')
    plt.show()
# %%


#######whole PSD 

# Define subjects and conditions
subjects = ['005']
conditions = ['LSD', 'PLA']
base_path = '/users/local/Venkatesh/LSD_project/src_data/derivatives/func/Video'

# Define canonical frequency bands
freq_bands = {
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 50)
}

whole_psd = {}

for condition in conditions:
    psd_data_all_subjects = []
    for subj in subjects:
        file_path = f'{base_path}/{condition}/sub-{subj}/meg/sub-{subj}_cleaned_epochs_meg.fif'
        
        # Read epochs
        epochs = mne.read_epochs(file_path)
        psd = epochs.compute_psd(fmax=61)
        psd_data = np.mean(psd.get_data(), axis=0)
        psd_data_db = 10 * np.log10(psd_data)
        
        psd_data_all_subjects.append(psd_data_db)
    
    whole_psd[condition] = np.array(psd_data_all_subjects)

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

def plot_psd(whole_psd_array):

    LSD_array_channel_avg = np.mean(whole_psd_array['LSD'], axis=1)  # Shape: (Subjects, Frequencies)
    PLA_array_channel_avg = np.mean(whole_psd_array['PLA'], axis=1)  # Shape: (Subjects, Frequencies)

    # Prepare DataFrame for seaborn
    freqs = np.arange(LSD_array_channel_avg.shape[1])/2  # Frequency bins
    df_LSD = pd.DataFrame(LSD_array_channel_avg, columns=freqs)
    df_LSD = df_LSD.melt(var_name='Frequency', value_name='Power')
    df_LSD['Condition'] = 'LSD'

    df_PLA = pd.DataFrame(PLA_array_channel_avg, columns=freqs)
    df_PLA = df_PLA.melt(var_name='Frequency', value_name='Power')
    df_PLA['Condition'] = 'PLA'

    # Combine the data
    df = pd.concat([df_LSD, df_PLA])

    # Plot using seaborn
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='Frequency', y='Power', hue='Condition', ci=95)
    plt.title('LSD and PLA PSD with 95% Confidence Intervals')
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.legend(title='Condition')
    plt.grid(True)
    plt.show()

# %%

####WHOLE PSD source level


import numpy as np
from scipy.signal import welch
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt

from mne.time_frequency import psd_array_multitaper
conditions = ['LSD', 'PLA']

# Parameters
fs = 250  # Sampling frequency (Hz)
n_permutations = 1000  # Number of permutations
freq_bands = {
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 100),
}
subjects = [ '005']

# Placeholder for PSD and statistics
whole_psd_source = {}

for condition in conditions:
    psd_subjects = []
    for subj in subjects:
        # Load parcellated source-level signal (shape: [n_epochs, n_regions, n_samples])
        file_path = f'{base_path}/{condition}/sub-{subj}/meg/source_estimates/sub-{subj}_source_estimate_parcellated.npz'
        source_data = np.load(file_path)['stc_parcellated']  # Load the parcellated data

        psd_epochs = []
        for epoch_data in source_data:  # Iterate over epochs (shape: [n_regions, n_samples])
            psd_region = list()
            for region_idx in range(epoch_data.shape[0]):  # Iterate over regions
                region_signal = epoch_data[region_idx]  # Shape: (n_samples,)
                
                psd, f = psd_array_multitaper(region_signal, sfreq=fs, fmin=1, fmax=60)
                
                psd_db = 10 * np.log10(psd)
                
                psd_region.append(psd_db)
            
            psd_epochs.append(np.array(psd_region))
            
        psd_subjects.append(np.mean(psd_epochs, axis=0))
        
    whole_psd_source[condition] = np.array(psd_subjects)
            
                
                
# %%
import matplotlib.pyplot as plt
# plt.plot(whole_psd_source['PLA'][0].T, color='red')
plt.plot(whole_psd_source['LSD'][0].T, color='blue', alpha=0.1)
# %%
plot_psd(whole_psd)
# %%
whole_psd_source['LSD'][0].shape
# %%
plt.plot(whole_psd['LSD'][0].T)
# %%

import numpy as np
from scipy.signal import welch
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt

from mne.time_frequency import psd_array_multitaper
conditions = ['LSD', 'PLA']

# Parameters
fs = 250  # Sampling frequency (Hz)
n_permutations = 1000  # Number of permutations
freq_bands = {
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 100),
}

# Placeholder for PSD and statistics
whole_psd_source = {}
subjects = [ '006']
conditions = ['LSD', 'PLA']
base_path = '/users/local/Venkatesh/LSD_project/src_data/derivatives/func/Video'

for condition in conditions:
    psd_subjects = []
    for subj in subjects:
        # Load parcellated source-level signal (shape: [n_epochs, n_regions, n_samples])
        file_path = f'{base_path}/{condition}/sub-{subj}/meg/source_estimates/sub-{subj}_source_estimate_parcellated.npz'
        source_data = np.load(file_path)['stc_parcellated']  # Load the parcellated data

        ch_array = [f"ch {i}" for i in range(360)]

        info = mne.create_info(ch_names=ch_array, sfreq=250, ch_types='mag')

        source_epochs= mne.EpochsArray(source_data, info)
        source_epochs.compute_psd(fmin=1, fmax=60).plot()
        plt.show()
# %%


plt.plot(psd[1], psd[0][0].T)
# %%


# %%
source_epochs.compute_psd(fmin=1, fmax=60).get_data().shape
# %%
