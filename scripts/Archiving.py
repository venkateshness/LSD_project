import tarfile
import argparse
import os

# Define the argument parser
parser = argparse.ArgumentParser(description='Zip a file to a specified directory.')
parser.add_argument('files', type=str, help='Path to the files')
parser.add_argument('zip', type=str, help='Directory to zip files to')

# Parse the arguments
args = parser.parse_args()


# Create a tar.gz archive
with tarfile.open(args.zip, "w:gz") as tar:
    tar.add(args.files, arcname=os.path.basename(args.files))

# Example usage:
# folder_to_tar = '/Brain/private/v20subra/LSD_project/src_data/derivatives/anat/'  # Change this to the path of the folder you want to tar
# tar_output = '/Brain/private/v20subra/LSD_project/src_data/derivatives/anat/pre_coreg.tar.gz'  # Change this to your desired output file name

# Uncomment the following line and modify the paths to use the function:
# tar_folder(folder_to_tar, tar_output)

# # %%

# %%
