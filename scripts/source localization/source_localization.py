


# Source space
# bem solution
# forward model
#inverse operator

import mne
import argparse
from joblib import Parallel, delayed
import os
import logging
import numpy as np
HOMEDIR = "/users/local/Venkatesh/LSD_project"
def create_source_space(subjects_dir, subject, drug):
    """Create the source space."""
    
    source_space_file = f"{subjects_dir}/anat/{drug}/{subject}/bem/{subject}-ico5-src.fif"
    subjects_dir = f"{subjects_dir}/anat/{drug}"
    
    if os.path.exists(source_space_file):
        print(f"Source space found for subject {subject}. Loading source space...")
        src = mne.read_source_spaces(source_space_file)
        print(f"Source space loaded for subject {subject}.")
        return src
    
    else:
        print(f"Creating source space for subject {subject}...")
        src = mne.setup_source_space(subject, spacing='ico5', subjects_dir=subjects_dir, n_jobs=-1)
        print(f"Source space setup complete for subject {subject}.")
        src_file = f"{subjects_dir}/{subject}/bem/{subject}-ico5-src.fif"
        mne.write_source_spaces(src_file, src, overwrite=True)
        
        print(f"Source space saved at {src_file}.")
    return src

def bem(subjects_dir, subject, drug):
    """Create or load the BEM solution."""
    bem_file = f"{subjects_dir}/anat/{drug}/{subject}/bem/{subject}-bem-sol.fif"
    subjects_dir = f"{subjects_dir}/anat/{drug}"
    
    if os.path.exists(bem_file):
        print(f"BEM solution found for subject {subject}. Loading BEM solution...")
        bem_sol = mne.read_bem_solution(bem_file)
        print(f"BEM solution loaded for subject {subject}.")
        return bem_sol
    
    else:
        print(f"BEM solution not found for subject {subject}. Creating BEM solution...")
        model = mne.make_bem_model(subject=subject, ico=5, conductivity=(0.3,), subjects_dir=subjects_dir)
        bem_sol = mne.make_bem_solution(model)
        mne.write_bem_solution(bem_file, bem_sol, overwrite=True)
        print(f"BEM solution created and saved at {bem_file}.")
        
        return bem_sol

def forward_model(subjects_dir, subject, epochs, trans, src, bem_sol, drug):
    """Create or load the forward model."""
    fwd_file = f"{subjects_dir}/anat/{drug}/{subject}/bem/{subject}-fwd.fif"
    subjects_dir = f"{subjects_dir}/anat/{drug}"

    if os.path.exists(fwd_file):
        print(f"Forward model found for subject {subject}. Loading forward model...")
        fwd = mne.read_forward_solution(fwd_file)
        print(f"Forward model loaded for subject {subject}.")
        return fwd
    
    else:
        print(f"Forward model not found for subject {subject}. Creating forward model...")
        fwd = mne.make_forward_solution(epochs.info, trans=trans, src=src, bem=bem_sol,
                                        meg=True, eeg=False)
        
        mne.write_forward_solution(fwd_file, fwd, overwrite=True)
        print(f"Forward model created and saved at {fwd_file}.")    
        return fwd


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


def parcellation(src):
    labels_parc = mne.read_labels_from_annot('fsaverage', parc='', subjects_dir=subjects_dir)

    label_ts = mne.extract_label_time_course(
    [stc], labels_parc, src, mode="mean", allow_empty=True
)

def morph_subject_activity_to_fsaverage(stcs, fwd, subject_from, subjects_dir, task, drug):
    """Morph the source estimate to fsaverage."""
    
    subjects_dir_anat = f"{subjects_dir}/anat/{drug}"
    
    src_to = mne.read_source_spaces(f"{subjects_dir_anat}/fsaverage/bem/fsaverage-ico-5-src.fif")    
    fsave_vertices = [s["vertno"] for s in src_to]
    stc_morphed_all_epochs = []
    morph = mne.compute_source_morph(fwd['src'], subject_to='fsaverage', src_to = src_to, spacing=fsave_vertices,
                                    
                                        subjects_dir= subjects_dir_anat)
    for idx, stc in enumerate(stcs):
        stc_morphed = morph.apply(stc)
        stc_morphed_all_epochs.append(stc_morphed)
    
    stc_morphed_all_epochs = np.array(stc_morphed_all_epochs)
    return stc_morphed_all_epochs
    #np.savez_compressed(f"{subjects_dir}/func/{task}/{drug}/{subject_from}/meg/source_estimates/{subject_from}_source_estimate_fsaverage.npz", stc_morphed_all_epochs)
    
    


def run_source_localization(subjects_dir, subject, task, drug):
    # Construct paths based on the subject
    subject = f"sub-{subject}"
    trans_file = f"{subjects_dir}/anat/{drug}/{subject}/bem/{subject}_trans.fif"
    trans = mne.read_trans(trans_file)
    epochs_file = f"{subjects_dir}/func/{task}/{drug}/{subject}/meg/{subject}_cleaned_epochs_meg.fif"

    epochs = mne.read_epochs(epochs_file, preload=True)
    epochs = epochs.pick_types(meg=True, eeg=False, ref_meg=False)
    
    if not os.path.exists(f"{subjects_dir}/func/{task}/{drug}/{subject}/meg/{subject}_cleaned_epochs_resampled_meg.fif"):
        epochs = epochs.resample(500)
        epochs.save(f"{subjects_dir}/func/{task}/{drug}/{subject}/meg/{subject}_cleaned_epochs_resampled_meg.fif", overwrite=True)
    
    else:
        epochs = mne.read_epochs(f"{subjects_dir}/func/{task}/{drug}/{subject}/meg/{subject}_cleaned_epochs_resampled_meg.fif", preload=True)
        
    # Create the source space
    src = create_source_space(subjects_dir, subject, drug)
    
    # Create or load the BEM solution
    bem_sol = bem(subjects_dir, subject, drug)

    # # Create or load the forward model
    fwd_model = forward_model(subjects_dir, subject, epochs, trans, src, bem_sol, drug)

    #Compute the noise covariance matrix
    noise_cov_data = np.eye(epochs.info['nchan']) 
    noise_cov = mne.Covariance(data=noise_cov_data, names=epochs.info['ch_names'], bads=[], projs=[], nfree=1)
    
    # Create the inverse operator
    inverse_operator = mne.minimum_norm.make_inverse_operator(epochs.info, fwd_model, noise_cov, loose=0.2, depth=0.8)
    print(f"Inverse operator created for subject {subject}.")

    # # Apply the inverse solution to create a source estimate
    method = "dSPM"  # could choose MNE, sLORETA, or eLORETA instead
    snr = 1.0 # or 1 
    lambda2 = 1.0 / snr**2
    stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inverse_operator, lambda2,
                                              method=method)
    print(f"Source localization complete for subject {subject}.")

    morphed_stc = morph_subject_activity_to_fsaverage(stcs, fwd_model, subject, subjects_dir, task, drug)    

    stc_data_all = []
    for stc in morphed_stc:
        stc_data_all.append(stc.data)
    
    stc_data_all = np.array(stc_data_all)
    if not os.path.exists(f"{subjects_dir}/func/{task}/{drug}/{subject}/meg/source_estimates"):
        os.makedirs(f"{subjects_dir}/func/{task}/{drug}/{subject}/meg/source_estimates")
    
    
    stc_parcellated_all = []
    for i in range(len(stc_data_all)):
        stc_parcellated = averaging_by_parcellation(stc_data_all[i])
        stc_parcellated_all.append(stc_parcellated)
    
    stc_parcellated_all = np.array(stc_parcellated_all)
    np.savez_compressed(f"{subjects_dir}/func/{task}/{drug}/{subject}/meg/source_estimates/{subject}_source_estimate_parcellated.npz", stc_parcellated = stc_parcellated_all)
    
    # np.savez_compressed(f"{subjects_dir}/func/{task}/{drug}/{subject}/meg/source_estimates/{subject}_source_estimate.npz", stc=stc_data_all)
    
    
    

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description='Perform source localization for multiple subjects in parallel using joblib.')
    # parser.add_argument('subjects_dir', type=str, help='Path to the subjects directory')
    parser.add_argument('-s', '--subjects', type=str, required=True, nargs='+', help='List of subject IDs to preprocess')
    parser.add_argument('-t', '--task', type=str, required=True, help='Task name (e.g., music, video)')
    parser.add_argument('-d', '--drug', type=str, required=True, help='Drug condition (e.g., LSD, PLA)')
    
    with np.load(
        f"{HOMEDIR}/src_data/sourcespace_to_glasser_labels.npz"
    ) as dobj:  # shoutout to https://github.com/rcruces/2020_NMA_surface-plot.git
        atlas = dict(**dobj)

    
    subjects_dir = '/users/local/Venkatesh/LSD_project/src_data/derivatives/'
    args = parser.parse_args()

    # Run subjects in parallel using joblib
    Parallel(n_jobs=1)(delayed(run_source_localization)(subjects_dir, subject, args.task, args.drug) for subject in args.subjects)


#%%

# regularization parameter
# MRI 
# Identity matrix
# oct 6 - comp cheaper
# fixed or Loose orientation constraint - 


#%%

# import mne
# import numpy as np

# rand = np.random.rand(20484, 2)

# stc = mne.SourceEstimate(data=rand, vertices=[np.arange(10242), np.arange(10242)], tmin=0, tstep=1/500)

# HCPMMP1_combined = mne.read_labels_from_annot('fsaverage', 'HCPMMP1_combined', subjects_dir='/users/local/Venkatesh/LSD_project/src_data/derivatives/anat/LSD')
# # %%
# subjects_dir = '/users/local/Venkatesh/LSD_project/src_data/derivatives/anat/LSD/'
# src = mne.setup_source_space(
#     'fsaverage', spacing="ico5", add_dist=False, subjects_dir=subjects_dir
# )
# labels=mne.read_labels_from_annot('fsaverage', 'HCPMMP1', subjects_dir='/users/local/Venkatesh/LSD_project/src_data/derivatives/anat/LSD')
# label_ts = mne.extract_label_time_course(
#     [stc], labels=labels, src=src, mode="mean", allow_empty=True
# )
# # %%
# label_ts
# # %%
morphedstc = np.load('/users/local/Venkatesh/LSD_project/src_data/derivatives/func/Music/PLA/sub-009/meg/source_estimates/sub-009_source_estimate_parcellated.npz', allow_pickle=True)['stc_parcellated']

from nilearn.regions import signals_to_img_labels
from nilearn.datasets import fetch_icbm152_2009
from nilearn import plotting
HOMEDIR = "/users/local/Venkatesh/LSD_project"
path_Glasser = f"{HOMEDIR}/src_data/Glasser_masker.nii.gz"
mnitemp  = fetch_icbm152_2009()
data_to_plot_morphed = np.mean(morphedstc, axis=(0,2))
zscore = (data_to_plot_morphed - np.mean(data_to_plot_morphed))/np.std(data_to_plot_morphed)
nifti= signals_to_img_labels(zscore, path_Glasser, mnitemp["mask"])

plotting.plot_img_on_surf(stat_map=nifti, views=["lateral", "medial"], hemispheres=["left", "right"], symmetric_cbar=True)

# %%
