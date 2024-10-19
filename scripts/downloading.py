#%%
import requests

import requests
import tarfile
import os
import pandas as pd
import numpy as np

# %%

df = pd.read_excel("/users/local/Venkatesh/LSD_project/src_data/IDs.xlsx", header=2)
id = df['ID'].values
# id_cleaned = np.array([x for x in id if x is not np.nan])

# id_cleaned = np.char.replace(id_cleaned, '_', '-')


def reverse_date_format(date_str):
    # Extract month, day, and year from the input string
    month = date_str[:2]  # First 2 characters: '01'
    day = date_str[2:4]   # Next 2 characters: '05'
    year = date_str[4:6]  # Last 2 characters: '14'

    # Transform to YYYYMMDD format
    transformed_date = '20' + year + day+month   # Concatenate '20' (assuming 21st century), year, month, and day
    
    return transformed_date


def download_file(url, output_path):
    """Download a file from a URL and save it locally."""
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(output_path, 'wb') as file:
            for chunk in response.iter_content(1024):
                if chunk:
                    file.write(chunk)
        print(f"Downloaded successfully: {output_path}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

def extract_tar_file(tar_path, extract_to):
    """Extract a .tar.gz file to a specific directory."""
    if tarfile.is_tarfile(tar_path):
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=extract_to)
            print(f"Extracted to {extract_to}")
    else:
        print(f"{tar_path} is not a valid .tar.gz file")

to_download = ['https://dl.dropboxusercontent.com/scl/fo/wqxnbthlftupwe5xxgvq1/h/070814-1_LSD_20140807_Video2.ds.tar.gz?rlkey=e72s8nq5liashuan7sm2mwpzj&dl=0'] # this one subject had data missing (HPI missing). They seem to have performed another scan.
 
# https://www.dropbox.com/scl/fo/wqxnbthlftupwe5xxgvq1/h/010514-1_LSD_20140501_Music.ds.tar.gz?rlkey=e72s8nq5liashuan7sm2mwpzj&dl=0

#to_download = ['https://dl.dropboxusercontent.com/scl/fo/wqxnbthlftupwe5xxgvq1/h/010813-4_LSD_20140820_Video.ds.tar.gz?rlkey=e72s8nq5liashuan7sm2mwpzj&dl=0',
#  'https://dl.dropboxusercontent.com/scl/fo/wqxnbthlftupwe5xxgvq1/h/010814-4_LSD_20140903_Video.ds.tar.gz?rlkey=e72s8nq5liashuan7sm2mwpzj&dl=0',
#  'https://dl.dropboxusercontent.com/scl/fo/wqxnbthlftupwe5xxgvq1/h/050913-1_LSD_20140529_Video.ds.tar.gz?rlkey=e72s8nq5liashuan7sm2mwpzj&dl=0',
#  'https://dl.dropboxusercontent.com/scl/fo/wqxnbthlftupwe5xxgvq1/h/050913-1_LSD_20140612_Video.ds.tar.gz?rlkey=e72s8nq5liashuan7sm2mwpzj&dl=0',
#  'https://dl.dropboxusercontent.com/scl/fo/wqxnbthlftupwe5xxgvq1/h/070814-1_LSD_20140807_Video.ds.tar.gz?rlkey=e72s8nq5liashuan7sm2mwpzj&dl=0',
#  'https://dl.dropboxusercontent.com/scl/fo/wqxnbthlftupwe5xxgvq1/h/070814-1_LSD_20140821_Video.ds.tar.gz?rlkey=e72s8nq5liashuan7sm2mwpzj&dl=0',
#  'https://dl.dropboxusercontent.com/scl/fo/wqxnbthlftupwe5xxgvq1/h/070814-2_LSD_20140807_Video.ds.tar.gz?rlkey=e72s8nq5liashuan7sm2mwpzj&dl=0',
#  'https://dl.dropboxusercontent.com/scl/fo/wqxnbthlftupwe5xxgvq1/h/070814-2_LSD_20140821_Video.ds.tar.gz?rlkey=e72s8nq5liashuan7sm2mwpzj&dl=0',
#  'https://dl.dropboxusercontent.com/scl/fo/wqxnbthlftupwe5xxgvq1/h/090714-1_LSD_20140709_Video.ds.tar.gz?rlkey=e72s8nq5liashuan7sm2mwpzj&dl=0',
#  'https://dl.dropboxusercontent.com/scl/fo/wqxnbthlftupwe5xxgvq1/h/090714-1_LSD_20140730_Video.ds.tar.gz?rlkey=e72s8nq5liashuan7sm2mwpzj&dl=0',
#  'https://dl.dropboxusercontent.com/scl/fo/wqxnbthlftupwe5xxgvq1/h/140514-1_LSD_20140514_Video.ds.tar.gz?rlkey=e72s8nq5liashuan7sm2mwpzj&dl=0',
#  'https://dl.dropboxusercontent.com/scl/fo/wqxnbthlftupwe5xxgvq1/h/140514-1_LSD_20140528_Video.ds.tar.gz?rlkey=e72s8nq5liashuan7sm2mwpzj&dl=0',
#  'https://dl.dropboxusercontent.com/scl/fo/wqxnbthlftupwe5xxgvq1/h/200814-1_LSD_20140820_Video.ds.tar.gz?rlkey=e72s8nq5liashuan7sm2mwpzj&dl=0',
#  'https://dl.dropboxusercontent.com/scl/fo/wqxnbthlftupwe5xxgvq1/h/200814-1_LSD_20140903_Video.ds.tar.gz?rlkey=e72s8nq5liashuan7sm2mwpzj&dl=0',
#  'https://dl.dropboxusercontent.com/scl/fo/wqxnbthlftupwe5xxgvq1/h/240107-6_LSD_20140626_Video.ds.tar.gz?rlkey=e72s8nq5liashuan7sm2mwpzj&dl=0',
#  'https://dl.dropboxusercontent.com/scl/fo/wqxnbthlftupwe5xxgvq1/h/240107-6_LSD_20140710_Video.ds.tar.gz?rlkey=e72s8nq5liashuan7sm2mwpzj&dl=0',
#  'https://dl.dropboxusercontent.com/scl/fo/wqxnbthlftupwe5xxgvq1/h/260614-1_LSD_20140626_Video.ds.tar.gz?rlkey=e72s8nq5liashuan7sm2mwpzj&dl=0',
#  'https://dl.dropboxusercontent.com/scl/fo/wqxnbthlftupwe5xxgvq1/h/260614-1_LSD_20140710_Video.ds.tar.gz?rlkey=e72s8nq5liashuan7sm2mwpzj&dl=0',
#  'https://dl.dropboxusercontent.com/scl/fo/wqxnbthlftupwe5xxgvq1/h/290514-2_LSD_20140529_Video.ds.tar.gz?rlkey=e72s8nq5liashuan7sm2mwpzj&dl=0',
#  'https://dl.dropboxusercontent.com/scl/fo/wqxnbthlftupwe5xxgvq1/h/290514-2_LSD_20140612_Video.ds.tar.gz?rlkey=e72s8nq5liashuan7sm2mwpzj&dl=0',
#  'https://dl.dropboxusercontent.com/scl/fo/wqxnbthlftupwe5xxgvq1/h/310714-1_LSD_20140731_Video.ds.tar.gz?rlkey=e72s8nq5liashuan7sm2mwpzj&dl=0',
#  'https://dl.dropboxusercontent.com/scl/fo/wqxnbthlftupwe5xxgvq1/h/310714-1_LSD_20140917_Video.ds.tar.gz?rlkey=e72s8nq5liashuan7sm2mwpzj&dl=0']


for i in to_download:
    dropbox_url = i
    print(dropbox_url)
    output_file = 'test.tar.gz'  # Local file name for downloaded file
    extract_dir = '/users/local/Venkatesh/LSD_project/src_data/ds_data/Video/'

    # Create directory to extract files if it doesn't exist
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)

    # Download and extract the file
    download_file(dropbox_url, output_file)
    extract_tar_file(output_file, extract_dir)

# %%
