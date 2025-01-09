#%%
import mne
from mne.coreg import Coregistration
import os
from joblib import Parallel, delayed

HOMEDIR = "/users/local/Venkatesh/LSD_project"

freesurfer_home = "/users/local/Venkatesh/LSD_project/src_data/Freesurfer/freesurfer"

def process_subjects(subject, subjects_dir):
        print(f"Processing subject {subject}..., from {subjects_dir}")
        
        mne.bem.make_watershed_bem(subject, subjects_dir=subjects_dir, overwrite=True)
        mne.bem.make_scalp_surfaces(subject, subjects_dir=subjects_dir, overwrite=True)


conditions = ['LSD', 'PLA']
for condition in conditions:
    subjects_dir = f"/users/local/Venkatesh/LSD_project/src_data/derivatives/anat/{condition}"
    subjects = os.listdir(f"{HOMEDIR}/src_data/derivatives/anat/{condition}/")
    subjects.remove('fsaverage')
    Parallel(n_jobs=-1)(delayed(process_subjects)(subject, subjects_dir) for subject in subjects)
    


#%%
import tarfile
import os

def tar_folder(folder_path, output_filename):
    """
    Create a tar.gz archive from a folder.
    
    :param folder_path: Path to the folder to archive.
    :param output_filename: Output filename for the tar.gz archive.
    """
    # Ensure the output filename ends with '.tar.gz'
    if not output_filename.endswith('.tar.gz'):
        output_filename += '.tar.gz'
        
    # Create a tar.gz archive
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(folder_path, arcname=os.path.basename(folder_path))

# Example usage:
folder_to_tar = '/users/local/Venkatesh/LSD_project/src_data/derivatives/anat/'  # Change this to the path of the folder you want to tar
tar_output = '/users/local/Venkatesh/LSD_project/src_data/derivatives/anat/pre_coreg.tar.gz'  # Change this to your desired output file name

# Uncomment the following line and modify the paths to use the function:
tar_folder(folder_to_tar, tar_output)

# # %%

# %%
