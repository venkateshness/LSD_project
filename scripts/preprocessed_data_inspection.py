#%%
import mne
import numpy as np

sub  = 'sub-08'
# data_mne_epochs = mne.read_epochs(f"/users/local/Venkatesh/LSD_project/src_data/derivatives/Music/LSD/{sub}/meg/{sub}_cleaned_epochs_meg.fif")

# bad_epochs_ar = np.load(f"/users/local/Venkatesh/LSD_project/src_data/derivatives/Music/LSD/{sub}/meg/{sub}_reject_log_AR_pre_meg.fif.npz")['bad_epochs']
raw = mne.io.read_raw_fif(f"/users/local/Venkatesh/LSD_project/src_data/fif_data_BIDS/Music/LSD/{sub}/ses-01/meg/{sub}_ses-01_task-Music_meg.fif", preload=True)
events = mne.events_from_annotations(raw)[0]
epochs = mne.Epochs(raw, events, tmin=0, tmax=2, baseline=None, preload=True, picks='meg')
#%%
epochs_filtered =epochs.filter(1, 125)


# %%

##Muscle artefact check
annot_all, score_all = [], []
from mne.preprocessing import annotate_muscle_zscore

for i, j in enumerate(data_mne_epochs.pick('mag')):
    info = data_mne_epochs[i].info
    make_raw = mne.io.RawArray(j, info, verbose=False)
    annot, score = annotate_muscle_zscore(make_raw, ch_type='mag', threshold=4, min_length_good=0.2, filter_freq=[110, 140], verbose=False)
    annot_all.append(annot)
    score_all.append(score)
#%%

sub  = 'sub-08'
conditions = ['Music', 'Video']
drug = ['LSD', 'PLA']


data_mne_epochs = mne.read_epochs(f"/users/local/Venkatesh/LSD_project/src_data/derivatives/Music/LSD/{sub}/meg/{sub}_cleaned_epochs_meg.fif")

data_mne_epochs.plot(picks=np.array(data_mne_epochs.ch_names[29:])[list(range(100))], n_channels=100)
data_mne_epochs.plot(picks=np.array(data_mne_epochs.ch_names[29:])[list(range(100, 200))], n_channels=100)
data_mne_epochs.plot(picks=np.array(data_mne_epochs.ch_names[29:])[list(range(200, 271))],n_channels=100)
# %%
subjects = [f'sub-{i:02d}' for i in range(1, 12)]
HOMEDIR = '/users/local/Venkatesh/LSD_project/src_data/derivatives/'
# Generate reports for each subject
drugs = ['LSD', 'PLA']
conditions = ['Music']
report = mne.Report(title=f'Visual Inspection of Preprocessed Data', verbose=True)
import os
for subject in subjects:
    for condition in conditions:
        for drug in drugs:
            
            input_file = f"{HOMEDIR}/{condition}/{drug}/{subject}/meg/{subject}_cleaned_epochs_meg.fif"
            if os.path.exists(input_file):
                epochs = mne.read_epochs(input_file, preload=True)

                report.add_figure(epochs.plot(picks=np.array(data_mne_epochs.ch_names[29:])[list(range(100))], n_channels=100), title=f'{subject} {condition} {drug}, first 100 channels')
                report.add_figure(epochs.plot(picks=np.array(data_mne_epochs.ch_names[29:])[list(range(100, 200))], n_channels=100), title=f'{subject} {condition} {drug}, channels 100-200')
                report.add_figure(epochs.plot(picks=np.array(data_mne_epochs.ch_names[29:])[list(range(200, 271))],n_channels=100), title=f'{subject} {condition} {drug}, channels 200-271')

        

# %%
# Save the report
output_report_file = f"{HOMEDIR}/Visual_inpection_preprocessed_data_{condition}.html"
report.save(output_report_file, overwrite=True, open_browser=False)
# %%
sub = 'sub-01'
raw = mne.io.read_raw_fif(f"/users/local/Venkatesh/LSD_project/src_data/fif_data_BIDS/Music/LSD/{sub}/ses-01/meg/{sub}_ses-01_task-Music_meg.fif", preload=True)

epochs_fixed_length = mne.make_fixed_length_epochs(raw, duration=2, preload=True)
epochs_fixed_length
# %%


raw.pick('meg').get_data().reshape(210, 300, 2400) == epochs_fixed_length.pick('meg').get_data()

# %%
from mne import make_fixed_length_events
events = make_fixed_length_events(raw, duration=2, overlap=0)
events

# %%
epochs_fixed_length.pick('meg').info['ch_names']
# %%
raw.pick('meg').get_data()
# %%
epochs_fixed_length.pick('meg').get_data()[-1]==raw.pick('meg').get_data()[:, 2400*209:]
# %%
