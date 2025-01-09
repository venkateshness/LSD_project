#%%
import mne
import pandas as pd

ids = pd.read_excel("/users/local/Venkatesh/LSD_project/src_data/IDs.xlsx", header=2)

for i in range(1, 12):
    i = f'{i:02d}'
    epochs_cleaned = mne.read_epochs(f"/users/local/Venkatesh/LSD_project/src_data/derivatives_old_sequence_idx/Video/PLA/sub-{i}/meg/sub-{i}_cleaned_epochs_meg.fif", preload=True, verbose=False)

    subject_id = epochs_cleaned.info['subject_info']['his_id'].replace('-', '_')
    print(f"Old ID: {i}")
    if subject_id == '010814_4':
        subject_id = '010813_4'
    print(int(ids['subject'][ids['ID']== subject_id]))
    
# %%
import numpy as np
task = 'Music'
drug = 'LSD'
sub = 'sub-011'
len(np.load(f'/users/local/Venkatesh/LSD_project/src_data/fif_data_BIDS_epochs/{task}/{drug}/{sub}/meg/{sub}_good_epochs_upon_visual_inspection_of_raw_filtered_epochs.npz')['arr_0'])
# %%
import mne

epochs = mne.read_epochs('/users/local/Venkatesh/LSD_project/src_data/derivatives/func/Music/LSD/sub-003/meg/sub-003_cleaned_epochs_meg.fif', preload=True)

epochs.info['nchan']
# %%
import numpy as np
np.eye(epochs.info['nchan'])
# %%
epochs = epochs.pick_types(meg=True, eeg=False, ref_meg=False)

# %%
epochs.info['nchan']
# %%
labels = mne.read_labels_from_annot('fsaverage', parc='HCPMMP1', subjects_dir='/homes/v20subra/mne_data/MNE-fsaverage-data/')

# %%
len(labels)
# %%
# Step 2: Read the HCP MMP parcellation labels from fsaverage
import mne
import numpy as np
labels = mne.read_labels_from_annot('fsaverage', parc='HCPMMP1', subjects_dir='/homes/v20subra/mne_data/MNE-fsaverage-data/')
src = mne.read_source_spaces('/homes/v20subra/mne_data/MNE-fsaverage-data/fsaverage/bem/fsaverage-vol-5-src.fif')
random_data = np.random.randn(20484, 100)
stc = mne.SourceEstimate(random_data, vertices=[np.arange(10242), np.arange(10242, 20484)], tmin=0., tstep=1.)
# Step 3: Extract the mean time course per label (ROI)
# Apply mean on the vertices to get a single signal per ROI


# %%

stc =mne.SourceEstimate(random_data, vertices=[np.arange(10242), np.arange(10242, 20484)], tmin=stc.tmin, tstep=stc.tstep)
# %%
import source_localization

def morph_subject_activity_to_fsaverage(stc, subject_from, subjects_dir, task, drug):
    """Morph the source estimate to fsaverage."""
    
    
    src_to = mne.read_source_spaces("/homes/v20subra/mne_data/MNE-fsaverage-data/fsaverage/bem/fsaverage-ico-5-src.fif")
    # subjects_dir = f"{subjects_dir}/anat"

    stc_morphed = mne.compute_source_morph(stc, subject_from=subject_from, src_to=src_to,
                                    subjects_dir=subjects_dir).apply(stc)
    
    



morph_subject_activity_to_fsaverage(stc, subject_from='sub-003', subjects_dir='/users/local/Venkatesh/LSD_project/src_data/derivatives/anat/', task='Music', drug='LSD')
# %%
stc
# %%
def morph_subject_activity_to_fsaverage(stc, subject_from, subjects_dir, task, drug):
    """Morph the source estimate to fsaverage."""
    
    subjects_dir_anat = f"{subjects_dir}/anat"
    
    src_to = mne.read_source_spaces(f"{subjects_dir_anat}/fsaverage/bem/fsaverage-ico-5-src.fif")
    
    
    stc_morphed = mne.compute_source_morph(stc, subject_from='sub-003', subject_to='fsaverage', src_to = src_to,
                                        subjects_dir= subjects_dir_anat).apply(stc)    
        
        
morph_subject_activity_to_fsaverage(stc, subject_from='sub-003', subjects_dir='/users/local/Venkatesh/LSD_project/src_data/derivatives/', task='Music', drug='LSD')
# %%
import mne

rh_stc = mne.read_source_estimate('/users/local/Venkatesh/LSD_project/src_data/derivatives/func/Music/LSD/sub-003/meg/sub-003_morphed-rh.stc')
# %%
rh_stc.mean().plot(subject='fsaverage', subjects_dir='/homes/v20subra/mne_data/MNE-fsaverage-data/', hemi='rh', backend='matplotlib', clim=dict(kind='percent', lims=[20, 50, 80]), spacing='ico5')
# %%
data = lh_stc.mean()
# %%
