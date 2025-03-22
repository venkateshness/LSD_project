#%%
import numpy as np
import mne
import scipy.stats
import os
os.chdir('/Brain/private/v20subra/LSD_project/scripts/')
import utility_functions
import scipy
import importlib
importlib.reload(utility_functions)


HOMEDIR = "/Brain/private/v20subra/LSD_project"

graph = scipy.io.loadmat(f'{HOMEDIR}/src_data/SC_avg56.mat')['SC_avg56']
laplacian, eigvals, eigvectors = utility_functions.eigmodes(graph)

#%%
#################################################

def load_graph_and_eigenmodes(graph_path):
    # Load SC matrix and compute laplacian, eigenvalues, and eigenvectors
    graph = scipy.io.loadmat(graph_path)['SC_avg56']
    laplacian, eigvals, eigvectors = utility_functions.eigmodes(graph)
    return laplacian, eigvals, eigvectors

def graph_psd_compute(src_data, eigvectors):
    # Compute PSD for each epoch and average across epochs
    n_epochs, _, _ = src_data.shape
    psd_all = list()
    for epoch in range(n_epochs):
        psd, _ = utility_functions.compute_gpsd(src_data[epoch], eigvectors)
        psd_all.append(psd)
    
    return np.mean(psd_all, axis=0)


    
def compute_psd_subjects(task, drug):
    psd_subjects = dict() 
    for sub in ['003', '005', '006', '009', '010', '013', '015', '016', '017', '018']:
        src_data = np.load(f'{HOMEDIR}/src_data/derivatives/func/{task}/{drug}/sub-{sub}/meg/source_estimates/sub-{sub}_.npz', allow_pickle=True)['stc_data_parcellated']
        psd_subjects[f'{sub}'] = graph_psd_compute(src_data, eigvectors)

    return psd_subjects

LSD = compute_psd_subjects(task='Music', drug='LSD')
PLA = compute_psd_subjects(task='Music', drug='PLA')
#%%

# LSD_normalized = np.array(list(LSD.values()))/np.max(np.array(list(LSD.values())), axis=1, keepdims=True)
# PLA_normalized = np.array(list(PLA.values()))/np.max(np.array(list(PLA.values())), axis=1, keepdims=True)
# %%
def frequency_splits(psd):
    low = psd[:, :50]
    medium = psd[:, 50:200]
    high = psd[:, 200:]
    return low, medium, high

LSD_low, LSD_medium, LSD_high = frequency_splits(np.array(list(LSD.values())))
PLA_low, PLA_medium, PLA_high = frequency_splits(np.array(list(PLA.values())))
# %%
import seaborn as sns
# sns.violinplot(data=[np.mean(LSD_low, axis=0), np.mean(PLA_low, axis=0)])
# sns.violinplot(data=[np.mean(LSD_medium, axis=0), np.mean(PLA_medium, axis=0)])
sns.violinplot(data=[np.mean(LSD_high, axis=0), np.mean(PLA_high, axis=0)])

#%%
# sns.violinplot(data=[np.mean(LSD_low, axis=0), np.mean(PLA_low, axis=0)])

# plt.hist((np.array(list(LSD.values()))[0] - np.min(np.array(list(LSD.values()))[0]))/(np.max(np.array(list(LSD.values()))[0])-np.min(np.array(list(LSD.values()))[0])))
# %%
scipy.stats.ttest_rel(np.mean(LSD_high, axis=0), np.mean(PLA_high, axis=0))
# %%
LSD_array = np.array(list(LSD.values()))
PLA_array = np.array(list(PLA.values()))



#%%
np.shape(LSD_high)

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

# Calculate the mean power spectral densities for each frequency band
LSD_means = [np.mean(LSD_low, axis=0), np.mean(LSD_medium, axis=0), np.mean(LSD_high, axis=0)]
PLA_means = [np.mean(PLA_low, axis=0), np.mean(PLA_medium, axis=0), np.mean(PLA_high, axis=0)]

# Prepare data for plotting in a tidy format
data = {
    'Frequency Band': ['Low', 'Medium', 'High'] * 2,
    'Condition': ['LSD'] * 3 + ['PLA'] * 3,
    'Mean PSD': LSD_means + PLA_means
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Create the plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Frequency Band', y='Mean PSD', hue='Condition', data=df, errorbar='se')

# Add statistical annotations (optional)
p_values = [
    ttest_rel(LSD_low.flatten(), PLA_low.flatten()).pvalue,
    ttest_rel(LSD_medium.flatten(), PLA_medium.flatten()).pvalue,
    ttest_rel(LSD_high.flatten(), PLA_high.flatten()).pvalue
]

# Annotate significance
for i, p_val in enumerate(p_values):
    x = i  # Position on the x-axis for each band
    y = max(LSD_means[i], PLA_means[i]) + 0.1  # Height for annotation above bar
    if p_val < 0.05:
        plt.text(x, y, '*', ha='center', va='bottom', color='black', fontsize=14)
    elif p_val < 0.01:
        plt.text(x, y, '**', ha='center', va='bottom', color='black', fontsize=14)
    elif p_val < 0.001:
        plt.text(x, y, '***', ha='center', va='bottom', color='black', fontsize=14)

# Customize and display plot
plt.title('Mean Power Spectral Density by Frequency Band and Condition')
plt.ylabel('Mean PSD')
plt.xlabel('Frequency Band')
plt.legend(title='Condition')
plt.tight_layout()
plt.show()

# %%
p_values
# %%
LSD_high_mean = np.mean(LSD_high, axis=1)
LSD_low_mean = np.mean(LSD_low, axis=1)
LSD_medium_mean = np.mean(LSD_medium, axis=1)

PLA_high_mean = np.mean(PLA_high, axis=1)
PLA_low_mean = np.mean(PLA_low, axis=1)
PLA_medium_mean = np.mean(PLA_medium, axis=1)
# Combine all data into a DataFrame
data = pd.DataFrame({
    
    'Condition': ['LSD'] * 30 + ['PLA'] * 30 ,
    'gFreq': ['Low'] * 10 + ['Med'] * 10 + ['High'] * 10 + ['Low'] * 10 + ['Med'] * 10 + ['High'] * 10,
   
    'gPSD': np.concatenate([LSD_low_mean, LSD_medium_mean,
                            LSD_high_mean, PLA_low_mean,
                            LSD_medium_mean, PLA_high_mean])
})

# %%
sns.barplot(x='gFreq', hue='Condition', y='gPSD', data=data)

# %%

import matplotlib.pyplot as plt
from scipy.stats import sem

plt.style.use('fivethirtyeight')
# Standard error of the mean (SEM)
sem_LSD = sem(LSD_array, axis=0)
sem_PLA = sem(PLA_array, axis=0)

plt.semilogy(eigvals, np.mean(LSD_array, axis=0))
plt.semilogy(eigvals, np.mean(PLA_array, axis=0))

mean_LSD = np.mean(LSD_array, axis=0)
mean_PLA = np.mean(PLA_array, axis=0)

# Plot means with semilog scale
plt.semilogy(eigvals, mean_LSD, label='Music LSD', color='blue')
plt.fill_between(eigvals, mean_LSD - sem_LSD, mean_LSD + sem_LSD, color='blue', alpha=0.3)

plt.semilogy(eigvals, mean_PLA, label='Music Placebo', color='orange')
plt.fill_between(eigvals, mean_PLA - sem_PLA, mean_PLA + sem_PLA, color='orange', alpha=0.3)

plt.xlabel('Eigenvalue')
plt.ylabel('graph Power Spectral Density')
plt.legend()
plt.grid(True,  linestyle='--', linewidth=0.5, alpha=0.7)

# %%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Convert data to long-form DataFrame for Seaborn
n_subjects = PLA_array.shape[0]
n_eigenvals = PLA_array.shape[1]

# Create a DataFrame
df = pd.DataFrame({
    "Eigenvalues": np.tile(eigvals, 2 * n_subjects),
    "Graph PSD": np.concatenate([PLA_array.flatten(), LSD_array.flatten()]),
    "Condition": np.repeat(["Placebo"] * n_subjects + ["LSD"] * n_subjects, n_eigenvals)
})

# Plot with Seaborn
plt.figure(figsize=(8,6))
sns.lineplot(
    data=df, 
    x="Eigenvalues", 
    y="Graph PSD", 
    hue="Condition", 
    style="Condition",
    lw=2,
    ci=95  # Automatically computes 95% confidence interval
)

# Log scale for better visualization
plt.yscale("log")
# plt.xscale("log")

# Labels and title
plt.xlabel("Graph Laplacian Eigenvalues")
plt.ylabel("Graph PSD")
plt.title("Graph PSD under LSD vs. Placebo (with 95% CI)")

# Grid and show
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.show()
#%%
LSD_array_energy = LSD_array*eigvals
PLA_array_energy = PLA_array*eigvals

# scipy.stats.ttest_rel(np.sum(LSD_array_energy,axis=1), np.sum(PLA_array_energy,axis=1))
sns.violinplot(data=[np.sum(LSD_array_energy,axis=1), np.sum(PLA_array_energy,axis=1)])

# %%
tasks = ['Music']
drugs = ['LSD', 'PLA']

sdi_all_PLA = dict()
sdi_all_LSD = dict()

for task in tasks:
    for drug in drugs:
        for sub in ['003', '005', '009', '010', '013', '015', '016', '018']:
            src_data = np.load(f'{HOMEDIR}/src_data/derivatives/func/{task}/{drug}/sub-{sub}/meg/source_estimates/sub-{sub}_source_estimate_parcellated.npz', allow_pickle=True)['stc_parcellated']
            src_data = src_data.reshape(src_data.shape[1], src_data.shape[2]* src_data.shape[0])
            sdi,_,_ = utility_functions.fullpipeline(envelope=src_data, eigevecs=eigvectors, eigvals=eigvals, is_surrogate=False, in_seconds=False)
            if drug == 'PLA':
                sdi_all_PLA[f'{sub}'] = sdi
            if drug == 'LSD':
                sdi_all_LSD[f'{sub}'] = sdi

# %%
t, p = scipy.stats.ttest_rel(np.array(list(sdi_all_LSD.values())), np.array(list(sdi_all_PLA.values())))
# %%

# %%
from nilearn import plotting
from nilearn.regions import signals_to_img_labels
from nilearn.datasets import fetch_icbm152_2009

path_Glasser = '/Brain/private/v20subra/LSD_project/src_data/Glasser_masker.nii.gz'
mnitemp = fetch_icbm152_2009()


tvals =t * (p<0.05)
nifti= signals_to_img_labels(tvals, path_Glasser, mnitemp["mask"])

plotting.plot_img_on_surf(stat_map=nifti, views=["lateral", "medial"], hemispheres=["left", "right"], symmetric_cbar=True, threshold=0.000001)
plt.show()
# %%
np.savez_compressed(f'/Brain/private/v20subra/LSD_project/src_data/derivatives/func/{tasks[0]}/group_level_SDI.npz',tvals=tvals)
# %%
LSD['003'].shape
# %%
