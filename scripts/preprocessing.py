#%%
import mne

data_mne = mne.io.read_raw_ctf("/users/local/Venkatesh/LSD_project/src_data/ds_data/LSD/240107-6_LSD_20140710_Music.ds", preload=True)
# %%
# %%
import pandas
df = pd.read_excel('/users/local/Venkatesh/LSD_project/src_data/LSD_PLA_order_ID.xlsx')

# %%
df
# %%
import numpy as np
import matplotlib.pyplot as plt
# %%
data_mne.notch_filter([50, 100, 150, 200, 250, 300, 350]).compute_psd(fmax=200).plot()
# %%
#notch filter - power line artefacts
# 

from mne.preprocessing import annotate_muscle_zscore

threshold_muscle = 5

annot, score = annotate_muscle_zscore(data_mne,
    ch_type="mag",
    threshold=threshold_muscle,
    min_length_good=0.1,
    filter_freq=[110, 140])
# %%
fig, ax = plt.subplots()
ax.plot(data_mne.times, score)
ax.axhline(y=threshold_muscle, color="r")
ax.set(xlabel="time, (s)", ylabel="zscore", title="Muscle activity")
# %%
order = np.arange(144, 164)
data_mne.set_annotations(annot)
data_mne.plot(start=5, duration=20, order=order)
# %%


import mne_bids_pipeline
# %%
mne.read_epochs('/users/local/Venkatesh/LSD_project/src_data/derivatives/LSD/sub-01/ses-01/meg/sub-01_ses-01_task-music_proc-clean_epo.fif')['Music_start'][0]
# %%
data_mne
# %%
