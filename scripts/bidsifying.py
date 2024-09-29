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
        
        # annotations = data_mne.annotations
        # annotations = mne.Annotations(onset=[0, 419.99], duration=[419.99, 0], description=['Music_start', 'Music_end'])
        # annotations.description = ["Music_start", "Music_end"]
        # data_mne.set_annotations(annotations)
        # events = np.array([[1, 0, 1],
        #                    [1200*419.99, 0, 2]])
        # event_id = {'Music_start':1, 'Music_end':2}
        
        annotations = data_mne.annotations
        annotations.description = ['Music_start']
        data_mne.set_annotations(annotations)
        
        
        # events, event_id = mne.events_from_annotations(data_mne)
        bids_path = BIDSPath(subject=f'{idx}', session='01', task='music', root=f'{HOMEDIR}/src_data/fif_data_BIDS/{task}/', datatype='meg')
        write_raw_bids(data_mne, bids_path=bids_path, overwrite=True, events=events, event_id=event_id)
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
# %%

data_mne.info.ch_names
# %%
# %%

# %%
import numpy as np
import mne
raw = data_mne
sfreq = raw.info['sfreq']  # Sampling frequency, e.g., 600 Hz
epoch_duration_sec = 2  # Duration of each epoch in seconds
epoch_duration_samples = int(epoch_duration_sec * sfreq)  # Duration in samples

n_samples = raw.n_times

event_times_samples = np.arange(0, n_samples, epoch_duration_samples)

event_times_sec = event_times_samples / sfreq

annotations = mne.Annotations(onset=event_times_sec, duration=epoch_duration_sec, description=['Music_start'] * len(event_times_sec))

raw.set_annotations(annotations)
#%%
