#%%
import os
import mne
import numpy as np
import matplotlib.pyplot as plt

HOMEDIR = '/users/local/Venkatesh/LSD_project/src_data/derivatives/'
subjects = [f'sub-{i:02d}' for i in range(1, 12)] 
occipital_channels = ['MLO51-3305', 'MZO03-3305', 'MRO51-3305', 'MLO42-3305', 'MLO41-3305']

alpha_band = (8, 13) 
drug = 'LSD'

group_psds = {condition: [] for condition in ['Music', 'Video']}

for subject in subjects:
    # Compute Power Spectral Density (PSD) for each condition
    conditions = ['Music', 'Video']
    for condition in conditions:
        
        input_file = os.path.join(HOMEDIR, f"{condition}/{drug}/{subject}/meg/", f'{subject}_cleaned_epochs_meg.fif')
        epochs = mne.read_epochs(input_file, preload=True)
        spectrum = mne.Epochs.compute_psd(epochs, fmin=0, fmax=250, picks=occipital_channels, method='multitaper', n_jobs=-1)
        group_psds[condition].append(np.mean(spectrum, axis=0))


# %%
# Plot individual subject changes between Music and Video conditions
plt.figure(figsize=(10, 6))
colors = plt.cm.Set3(np.linspace(0, 1, len(subjects)))
for i in range(len(subjects)):
    plt.plot([0, 1], [
        np.array(group_psds['Music'])[i, 0, 16:26].mean(),
        np.array(group_psds['Video'])[i, 0, 16:26].mean()
    ], marker='o', color=colors[i], linestyle='-', linewidth=1)

# Boxplot for group comparison
plt.boxplot([np.array(group_psds['Music'])[:, 0, 16:26].mean(axis=1), np.array(group_psds['Video'])[:, 0, 16:26].mean(axis=1)],
            positions=[0, 1], widths=0.5, patch_artist=True, showmeans=True,
            boxprops=dict(facecolor='lightcoral', color='black'),
            medianprops=dict(color='black'),
            meanprops=dict(marker='o', markerfacecolor='red', markeredgecolor='red'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'))
plt.xticks([0, 1], ['Music (eyes closed)', 'Video (eyes open)'])
plt.ylabel('Alpha Power (8-13 Hz)')
plt.title(f'{drug}')
plt.grid(axis='y')
plt.show()
plt.close()
# %%


