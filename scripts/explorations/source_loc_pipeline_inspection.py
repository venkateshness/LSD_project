#%%
import mne


src = mne.read_source_spaces('/users/local/Venkatesh/LSD_project/src_data/derivatives/anat/sub-003/bem/sub-003-ico5-src.fif')
# %%
subject = "sub-003"
subjects_dir = "/users/local/Venkatesh/LSD_project/src_data/derivatives/anat"
plot_bem_kwargs = dict(
    subject=subject,
    subjects_dir=subjects_dir,
    brain_surfaces="white",
    orientation="coronal",
    slices=[50, 100, 150, 200],
)

mne.viz.plot_bem(src=src, **plot_bem_kwargs)
# %%
