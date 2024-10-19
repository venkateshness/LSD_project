#%%

import mne
import os
import pandas as pd
from mne_bids import BIDSPath, write_raw_bids
import numpy as np

HOMEDIR = '/users/local/Venkatesh/LSD_project/'

#%%
drug_condition = ['LSD', 'PLA']
task_condition = ['Music'] #, 'Music'

ID_order = []

for task in task_condition:
    for drug in drug_condition:
        ds_directories = os.listdir(f'{HOMEDIR}/src_data/ds_data/{task}/{drug}')
        
        for i, directory in enumerate(ds_directories):
            idx = f'{i+1:02}'
            
            data_mne = mne.io.read_raw_ctf(f"{HOMEDIR}/src_data/ds_data/{task}/{drug}/{directory}")
            
            # output_dir = f"{HOMEDIR}/src_data/fif_data/{task}/{directory}/"
            # os.makedirs(output_dir, exist_ok=True)
            # data_mne.save(f"{output_dir}/{directory.split('.')[0]}.fif")

            # annotations = data_mne.annotations
            
            
            annotations = mne.Annotations(onset=[0], duration=[2], description=[f'{task}_epochs'])
            
            annotations.description = [f'{task}_epochs'] * len(annotations.onset)
            # annotations.duration = [2] 
            data_mne.set_annotations(annotations)
            
            # sfreq = data_mne.info['sfreq']  # Sampling frequency, e.g., 600 Hz
            # epoch_duration_sec = 2  # Duration of each epoch in seconds
            # epoch_duration_samples = int(epoch_duration_sec * sfreq)  # Duration in samples

            # n_samples = data_mne.n_times
            # events, event_id = mne.events_from_annotations(data_mne)
            # beginning = events[:, 0][0]

            # event_times_samples = np.arange(beginning, n_samples, epoch_duration_samples)
            # event_times_sec = event_times_samples / sfreq
            # annotations = mne.Annotations(onset=event_times_sec, duration=epoch_duration_sec, description=[f'{task}_epochs'] * len(event_times_sec))
            # data_mne.set_annotations(annotations)
            
            events = mne.make_fixed_length_events(data_mne, duration=2, overlap=0)
            event_id = {'Music_epochs': 1}
            
            
            # events, event_id = mne.events_from_annotations(data_mne)
            bids_path = BIDSPath(subject=f'{idx}', session='01', task=f'{task}', root=f'{HOMEDIR}/src_data/fif_data_BIDS/{task}/{drug}/')
            write_raw_bids(data_mne, bids_path=bids_path, overwrite=True, events=events, event_id=event_id, format="FIF")
            ID_order.append(directory)


# %%

for drug in drug_condition:
    participants = pd.read_csv(f'{HOMEDIR}/src_data/fif_data_BIDS/{drug}/participants.tsv', delimiter='\t')
    if drug=='LSD':
        participants['subID'] = ID_order[:11]
    if drug=='PLA':
        participants['subID'] = ID_order[11:]
    
    participants.to_csv(f'{HOMEDIR}/src_data/fif_data_BIDS/{drug}/participants_included.tsv')

# %%

