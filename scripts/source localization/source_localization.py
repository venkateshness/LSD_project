
import mne
import argparse
from joblib import Parallel, delayed
import os
import logging
import numpy as np
HOMEDIR = "/Brain/private/v20subra/LSD_project"
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


def parcellation(stc):
    """Parcellate the source estimate."""
    labels=mne.read_labels_from_annot('fsaverage', 'HCPMMP1', sort=False, subjects_dir='/Brain/private/v20subra/LSD_project/src_data/derivatives/anat/LSD')
    src = mne.read_source_spaces(f"{subjects_dir}/anat/LSD/fsaverage/bem/fsaverage-ico-5-src.fif")

    exclude_indices = [0, 181]
    valid_labels = [label for i, label in enumerate(labels) if i not in exclude_indices]
    
    label_ts = mne.extract_label_time_course(
        stc, labels=valid_labels, src=src, mode="mean", allow_empty=True, mri_resolution=False
    )
    return label_ts

def morph_subject_activity_to_fsaverage(stcs, fwd, subject_from, subjects_dir, task, drug):
    """Morph the source estimate to fsaverage."""
    
    subjects_dir_anat = f"{subjects_dir}/anat/{drug}"
    
    src_to = mne.read_source_spaces(f"{subjects_dir_anat}/fsaverage/bem/fsaverage-ico-5-src.fif")    
    fsave_vertices = [s["vertno"] for s in src_to]
    stc_morphed_all_epochs = []
    morph = mne.compute_source_morph(fwd['src'], subject_to='fsaverage', src_to = src_to, spacing=fsave_vertices,
                                    
                                        subjects_dir= subjects_dir_anat)
    for _, stc in enumerate(stcs):
        stc_morphed = morph.apply(stc)
        stc_morphed_all_epochs.append(stc_morphed)
    
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
        epochs = epochs.resample(250)
        epochs.save(f"{subjects_dir}/func/{task}/{drug}/{subject}/meg/{subject}_cleaned_epochs_resampled_meg.fif", overwrite=True)
    
    else:
        epochs = mne.read_epochs(f"{subjects_dir}/func/{task}/{drug}/{subject}/meg/{subject}_cleaned_epochs_resampled_meg.fif", preload=True)
        
    # # Create the source space
    src = create_source_space(subjects_dir, subject, drug)

    # Create or load the BEM solution
    bem_sol = bem(subjects_dir, subject, drug)

    #  Create or load the forward model
    fwd_model = forward_model(subjects_dir, subject, epochs, trans, src, bem_sol, drug)

    #Compute the noise covariance matrix
    noise_cov_data = np.eye(epochs.info['nchan']) 
    noise_cov = mne.Covariance(data=noise_cov_data, names=epochs.info['ch_names'], bads=[], projs=[], nfree=1)    
    
    # Create the inverse operator
    inverse_operator = mne.minimum_norm.make_inverse_operator(epochs.info, fwd_model, noise_cov, loose=0.2, depth=0.8)
    print(f"Inverse operator created for subject {subject}.")

    # Apply the inverse solution to create a source estimate
    method = "dSPM"  # could choose MNE, sLORETA, or eLORETA instead
    snr = 3.0 # or 1 
    lambda2 = 1.0 / snr**2
    
    stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inverse_operator, lambda2,
                                                method=method)
    
    morphed_stc = morph_subject_activity_to_fsaverage(stcs, fwd_model, subject, subjects_dir, task, drug)    

    stc_data_parcellated = []
    for stc in morphed_stc:
        stc_data_parcellated.append(parcellation(stc))
        
    if not os.path.exists(f"{subjects_dir}/func/{task}/{drug}/{subject}/meg/source_estimates"):
        os.makedirs(f"{subjects_dir}/func/{task}/{drug}/{subject}/meg/source_estimates")
            
    np.savez_compressed(f"{subjects_dir}/func/{task}/{drug}/{subject}/meg/source_estimates/{subject}_.npz", stc_data_parcellated = stc_data_parcellated)
    

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

    
    subjects_dir = '/Brain/private/v20subra/LSD_project/src_data/derivatives/'
    args = parser.parse_args()

    # Run subjects in parallel using joblib
    Parallel(n_jobs=1)(delayed(run_source_localization)(subjects_dir, subject, args.task, args.drug) for subject in args.subjects)

