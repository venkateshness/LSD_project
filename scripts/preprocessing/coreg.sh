#!/bin/bash
# Source FreeSurfer setup
source /users/local/Venkatesh/LSD_project/src_data/Freesurfer/freesurfer/SetUpFreeSurfer.sh  # Replace with actual path to FreeSurfer

# Run the Python script
python /users/local/Venkatesh/LSD_project/scripts/coregistration.py #--subjects 01 02 03 04 06 09 10 11 12 13 15 17 18 19 20
