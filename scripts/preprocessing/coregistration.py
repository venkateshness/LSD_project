#%%
import mne
from mne.coreg import Coregistration
import os
from joblib import Parallel, delayed

HOMEDIR = "/Brain/private/v20subra/LSD_project"

freesurfer_home = "/Brain/private/v20subra/LSD_project/src_data/Freesurfer/freesurfer"

def process_subjects(subject, subjects_dir):
        print(f"Processing subject {subject}..., from {subjects_dir}")
        
        mne.bem.make_watershed_bem(subject, subjects_dir=subjects_dir, overwrite=True)
        mne.bem.make_scalp_surfaces(subject, subjects_dir=subjects_dir, overwrite=True)


conditions = ['LSD', 'PLA']
for condition in conditions:
    subjects_dir = f"/Brain/private/v20subra/LSD_project/src_data/derivatives/anat/{condition}"
    subjects = os.listdir(f"{HOMEDIR}/src_data/derivatives/anat/{condition}/")
    subjects.remove('fsaverage')
    Parallel(n_jobs=-1)(delayed(process_subjects)(subject, subjects_dir) for subject in subjects)
    


#%%
