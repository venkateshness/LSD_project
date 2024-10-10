#%%

import mne
import os
import pandas as pd
from mne_bids import BIDSPath, write_raw_bids
import numpy as np

HOMEDIR = '/users/local/Venkatesh/LSD_project/'

#%%
task_conditions = ['LSD', 'PLA']
ID_order = []
for task in task_conditions:
    ds_directories = os.listdir(f'{HOMEDIR}/src_data/ds_data/{task}')
    
    for i, directory in enumerate(ds_directories):
        idx = f'{i+1:02}'
        
        data_mne = mne.io.read_raw_ctf(f"{HOMEDIR}/src_data/ds_data/{task}/{directory}")
        
        # output_dir = f"{HOMEDIR}/src_data/fif_data/{task}/{directory}/"
        # os.makedirs(output_dir, exist_ok=True)
        # data_mne.save(f"{output_dir}/{directory.split('.')[0]}.fif")

        
        # events, event_id = mne.events_from_annotations(data_mne)
        bids_path = BIDSPath(subject=f'{idx}', session='01', task='music', root=f'{HOMEDIR}/src_data/fif_data_BIDS/{task}/')
        write_raw_bids(data_mne, bids_path=bids_path, overwrite=True, format="FIF")
        ID_order.append(directory)


# %%

for task in task_conditions:
    participants = pd.read_csv(f'{HOMEDIR}/src_data/fif_data_BIDS/{task}/participants.tsv', delimiter='\t')
    if task=='LSD':
        participants['subID'] = ID_order[:11]
    if task=='PLA':
        participants['subID'] = ID_order[11:]
    
    participants.to_csv(f'{HOMEDIR}/src_data/fif_data_BIDS/{task}/participants_included.tsv')

# %%
