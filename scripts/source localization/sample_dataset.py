

import matplotlib.pyplot as plt
import numpy as np

import mne
from mne.datasets import sample
from mne.minimum_norm import apply_inverse, make_inverse_operator

# %%
# Process MEG data

data_path = sample.data_path()
raw_fname = data_path / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif"

raw = mne.io.read_raw_fif(raw_fname)  # already has an average reference
events = mne.find_events(raw, stim_channel="STI 014")

event_id = dict(aud_l=1)  # event trigger and conditions
tmin = -0.2  # start of each epoch (200ms before the trigger)
tmax = 0.5  # end of each epoch (500ms after the trigger)
raw.info["bads"] = ["MEG 2443", "EEG 053"]  # mark known bad channels
baseline = (None, 0)  # means from the first instant to t = 0
reject = dict(grad=4000e-13, mag=4e-12, eog=150e-6)

epochs = mne.Epochs(
    raw,
    events,
    event_id,
    tmin,
    tmax,
    proj=True,
    picks=("meg", "eog"),
    baseline=baseline,
    reject=reject,
)

# # Create the source space
src = create_source_space(subjects_dir, subject, drug)

# Create or load the BEM solution
bem_sol = bem(subjects_dir, subject, drug)

# # # Create or load the forward model
fwd_model = forward_model(subjects_dir, subject, epochs, None, src, bem_sol, drug)

#Compute the noise covariance matrix
noise_cov_data = np.eye(epochs.info['nchan']) 
noise_cov = mne.Covariance(data=noise_cov_data, names=epochs.info['ch_names'], bads=[], projs=[], nfree=1)

# Create the inverse operator
inverse_operator = mne.minimum_norm.make_inverse_operator(epochs.info, fwd_model, noise_cov, loose=0.2, depth=0.8)
print(f"Inverse operator created for subject {subject}.")

# # # Apply the inverse solution to create a source estimate
method = "dSPM"  # could choose MNE, sLORETA, or eLORETA instead
snr = 3.0 # or 1 
lambda2 = 1.0 / snr**2
evoked = epochs.average()
# stcs = mne.minimum_norm.apply_inverse(evoked, inverse_operator, lambda2,
#                                           method=method)`
stc, residual = mne.minimum_norm.apply_inverse(
    evoked,
    inverse_operator,
    lambda2,
    method=method,
    pick_ori=None,
    return_residual=True,
    verbose=True,
)

# source_psd = mne.minimum_norm.compute_source_psd_epochs(epochs[:2], inverse_operator, lambda2=lambda2, method=method, fmin=1.0, fmax=60.0, pick_ori=None, label=None, nave=1, pca=True, inv_split=None, adaptive=False, low_bias=True, return_generator=False, n_jobs=None, prepared=False, method_params=None, return_sensor=False, use_cps=True, verbose=None)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()

plt.plot(1e3 * stc.times, stc.data[::100, :].T)

# ax.set(xlabel="time (ms)", ylabel=f"{method} value")

plt.show()


vertno_max, time_max = stc.get_peak(hemi="rh")

subjects_dir = "/users/local/Venkatesh/LSD_project/src_data/derivatives/anat/LSD"
surfer_kwargs = dict(
    hemi="rh",
    subjects_dir=subjects_dir,
    clim=dict(kind="value", lims=[8, 12, 15]),
    
    initial_time=time_max,
    time_unit="s",
    size=(800, 800),
    smoothing_steps=15,
)
brain = stc.plot(**surfer_kwargs, backend='matplotlib')
brain.add_foci(
    vertno_max,
    coords_as_verts=True,
    hemi="rh",
    color="blue",
    scale_factor=0.6,
    alpha=0.5,
)
brain.add_text(
    0.1, 0.9, "dSPM (plus location of maximal activation)", "title", font_size=14
)

# The documentation website's movie is generated with:
# brain.save_movie(..., tmin=0.05, tmax=0.15, interpolation='linear',
#                  time_dilation=20, framerate=10, time_viewer=True)