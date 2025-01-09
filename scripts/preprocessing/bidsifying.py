#%%

import mne
import os
import pandas as pd
from mne_bids import BIDSPath, write_raw_bids
import numpy as np

HOMEDIR = '/users/local/Venkatesh/LSD_project/'

#%%
drug_condition = ['LSD', 'PLA']
task_condition = ['RestO', 'RestC'] #, 'Music'

ID_order = []

ID_order = []
ID_file = pd.read_excel(f'{HOMEDIR}/src_data/IDs.xlsx', header=2)
for task in task_condition:
    for drug in drug_condition:
        ds_directories = sorted(os.listdir(f'{HOMEDIR}/src_data/ds_data/{task}/{drug}'))

        
        for i, directory in enumerate(ds_directories):
            subj_ID = directory[:8]
            subj_ID = subj_ID.replace('-', '_')
            
            idx = int(ID_file['subject'][ID_file['ID'] == subj_ID].values[0])
            idx = f'{idx:03d}'

            data_mne = mne.io.read_raw_ctf(f"{HOMEDIR}/src_data/ds_data/{task}/{drug}/{directory}")            
            bids_path = BIDSPath(subject=f'{idx}', session='01', task=f'{task}', root=f'{HOMEDIR}/src_data/fif_data_BIDS/{task}/{drug}/')
            write_raw_bids(data_mne, bids_path=bids_path, overwrite=True, format="FIF")
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


############change FIF_data_BIDS_epochs folder to the canonical ordering of subjs

import os
import mne
import pandas as pd

tasks = ['Video']
drugs = ['LSD', 'PLA']

for task in tasks:
    for drug in drugs:
        for sub in os.listdir(f'{HOMEDIR}/src_data/derivatives_old_sequence_idx/{task}/{drug}/'):
            
            epochs = mne.read_epochs(f'{HOMEDIR}/src_data/derivatives_old_sequence_idx/{task}/{drug}/{sub}/meg/{sub}_cleaned_epochs_meg.fif')
            epochs.info['subject_info']['his_id'] = epochs.info['subject_info']['his_id'].replace('-', '_')
            
            if subj_ID == '010814_4':
                epochs.info['subject_info']['his_id'] = subj_ID.replace('010814_4', '010813_4')
            subj_ID = epochs.info['subject_info']['his_id']
            
            ID_file = pd.read_excel(f'{HOMEDIR}/src_data/IDs.xlsx', header=2)
            idx = int(ID_file['subject'][ID_file['ID'] == subj_ID].values[0])
            idx = f'sub-{idx:03d}'
            os.rename(f'{HOMEDIR}/src_data/fif_data_BIDS_epochs/{task}/{drug}/{sub}/', f'{HOMEDIR}/src_data/fif_data_BIDS_epochs/{task}/{drug}/{idx}')

#%%

tasks = ['Music']
drugs = [ 'PLA']

for task in tasks:
    for drug in drugs:
        for sub in os.listdir(f'{HOMEDIR}/src_data/fif_data_BIDS_epochs/{task}/{drug}/'):
            
            old_name = os.listdir(f'{HOMEDIR}/src_data/fif_data_BIDS_epochs/{task}/{drug}/{sub}/meg/')
            new_name = f'{sub}_good_epochs_upon_visual_inspection_of_raw_filtered_epochs.npz'
            os.rename(f'{HOMEDIR}/src_data/fif_data_BIDS_epochs/{task}/{drug}/{sub}/meg/{old_name[0]}', f'{HOMEDIR}/src_data/fif_data_BIDS_epochs/{task}/{drug}/{sub}/meg/{new_name}')
            
# %%
mne.io.read_raw_fif('/users/local/Venkatesh/LSD_project/src_data/derivatives/func/Music/LSD/sub-003/meg/sub-003_raw_notch_meg.fif').info['subject_info']['his_id']
# %%
