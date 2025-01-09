#%%
import mne

epochs_cleaned = mne.read_epochs("/users/local/Venkatesh/LSD_project/src_data/derivatives/Music/LSD/sub-01/meg/sub-01_cleaned_epochs_meg.fif")
epochs_cleaned_resampled=epochs_cleaned.copy().resample(500)
# %%
epochs_cleaned[1]
epochs_cleaned_resampled[1].plot()
# %%
import matplotlib.pyplot as plt
plt.plot(epochs_cleaned[1].get_data()[0, 0])
plt.plot(epochs_cleaned_resampled[1].get_data()[0, 0])
# %%
plt.show()
# %%
epochs_cleaned_resampled.compute_psd(fmax=125).plot()

# %%
epochs_cleaned.compute_psd(fmax=125).plot()
# %%
