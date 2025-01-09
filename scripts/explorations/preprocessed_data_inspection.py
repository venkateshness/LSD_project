#%%
import mne
import numpy as np
from mne.preprocessing import annotate_muscle_zscore

sub  = 'sub-03'
data_mne_epochs = mne.read_epochs(f"/users/local/Venkatesh/LSD_project/src_data/derivatives/Music/LSD/{sub}/meg/{sub}_cleaned_epochs_meg.fif")

# bad_epochs_ar = np.load(f"/users/local/Venkatesh/LSD_project/src_data/derivatives/Music/LSD/{sub}/meg/{sub}_reject_log_AR_pre_meg.fif.npz")['bad_epochs']
raw = mne.io.read_raw_fif(f"/users/local/Venkatesh/LSD_project/src_data/fif_data_BIDS/Music/LSD/{sub}/ses-01/meg/{sub}_ses-01_task-Music_meg.fif", preload=True)
events = mne.events_from_annotations(raw)[0]
epochs = mne.Epochs(raw, events, tmin=0, tmax=2, baseline=None, preload=True, picks='meg')

filtered_data = mne.io.read_raw_fif(f"/users/local/Venkatesh/LSD_project/src_data/derivatives/Music/LSD/sub-03/meg/sub-03_raw_filtered_meg.fif", preload=True)
epochs_fixed_length = mne.make_fixed_length_epochs(filtered_data, duration=2, preload=True)

#%%
# epochs_fixed_length.drop_bad(reject_dict).plot()
import matplotlib.pyplot as plt
data= epochs_fixed_length.pick("mag").get_data()
zscore = (data - np.mean(data, keepdims=True)) / np.std(data, keepdims=True)
# plt.hist(zscore, bins=1000)


#%%
##Muscle artefact check
annot_all, score_all = [], []
filtered_data = mne.io.read_raw_fif(f"/users/local/Venkatesh/LSD_project/src_data/derivatives/Music/LSD/sub-03/meg/sub-03_raw_filtered_meg.fif", preload=True)
epochs_fixed_length = mne.make_fixed_length_epochs(filtered_data, duration=2, preload=True)


for i, j in enumerate(data_mne_epochs.pick('mag')):
    
    info = data_mne_epochs[i].info
    make_raw = mne.io.RawArray(j, info, verbose=False)
    
    annot, score = annotate_muscle_zscore(make_raw, ch_type='mag', threshold=5, min_length_good=0.2, filter_freq=[110, 140], verbose=False, n_jobs=-1)
    annot_all.append(annot)
    score_all.append(score)
#%%
# np.savez_compressed("/users/local/Venkatesh/LSD_project/src_data/derivatives/Music/LSD/sub-01/annot_muscle.npz", **dict(zip(annot_all[0], score_all)))
import matplotlib.pyplot as plt
np.sum(np.abs(score_all )>5, axis=1)


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


# %%
data_mne_epochs.plot(picks=np.array(data_mne_epochs.ch_names[29:])[list(range(100))], n_channels=100)
data_mne_epochs.plot(picks=np.array(data_mne_epochs.ch_names[29:])[list(range(100, 200))], n_channels=100)
data_mne_epochs.plot(picks=np.array(data_mne_epochs.ch_names[29:])[list(range(200, 271))],n_channels=100)
# %%

# %%
import mne
filtered_data = mne.io.read_raw_fif(f"/users/local/Venkatesh/LSD_project/src_data/derivatives/Music/LSD/sub-03/meg/sub-03_raw_filtered_meg.fif", preload=True)
epochs_fixed_length = mne.make_fixed_length_epochs(filtered_data, duration=2, preload=True)

# %%

###############
# %%

def find_optimal_threshold(subject_id,  channel_type='mag', start_threshold=1e-12, step=1e-12, max_threshold=1e-11):
    """
    Iteratively find the optimal threshold for rejecting epochs based on peak-to-peak amplitude.
    """
    raw = mne.io.read_raw_fif(f"/users/local/Venkatesh/LSD_project/src_data/fif_data_BIDS/Music/LSD/{subject_id}/ses-01/meg/{subject_id}_ses-01_task-Music_meg.fif", preload=True)
    epochs_fixed_length = mne.make_fixed_length_epochs(raw, duration=2, preload=True, proj=False)
    
    for threshold in [4e-12]:
        reject_criteria = {channel_type: threshold}
        rejected_epochs = epochs_fixed_length.copy().drop_bad(reject=reject_criteria)
        rejected_count = len(epochs_fixed_length) - len(rejected_epochs)

        # Check if the number of rejected epochs is minimized while still removing artifacts
      
        print(f"Threshold: {threshold}, Rejected epochs: {rejected_count}")
    

find_optimal_threshold('sub-02',  channel_type='mag')
# %%
epochs_fixed_length = mne.make_fixed_length_epochs(raw, duration=2, preload=True)

reject = {'mag': 4e-12}
epochs_fixed_length.drop_bad(reject).plot()
# %%
import os
import mne
import numpy as np
from mne.report import Report
import matplotlib.pyplot as plt

subjects = ["sub-01"]
HOMEDIR = '/users/local/Venkatesh/LSD_project/'
drugs = ['LSD']
conditions = ['Music']

# Initialize the report
report = Report(title='Visual Inspection of MEG Signal', verbose=True)

# Loop through each subject, condition, and drug
for subject in subjects:
    for condition in conditions:
        for drug in drugs:
            
            input_file = f"{HOMEDIR}/src_data/derivatives/{condition}/{drug}/{subject}/meg/{subject}_cleaned_epochs_meg.fif"
            
            epochs_fixed_length = mne.read_epochs(input_file, preload=True, verbose=False)
            
            # epochs_fixed_length = mne.make_fixed_length_epochs(raw, duration=2, preload=True)
            if os.path.exists(input_file):
                epochs = epochs_fixed_length
                
                # Plot the MEG signal for 50 channels per figure and 20 epochs per figure
                for start_ch in range(0, 271, 50):
                    for start_ep in range(0, 209, 20):
                        epochs = epochs.pick('mag')
                        fig = epochs.plot(
                            picks=np.arange(start_ch, min(start_ch + 50, len(epochs.ch_names))),
                            n_epochs=20,
                            n_channels=50,
                            show=False
                        )
                        report.add_figure(fig, title=f'{subject} {condition} {drug} - Channels {start_ch}-{start_ch + 50}, Epochs {start_ep}-{start_ep + 20}')
                        plt.close(fig)
# Save the report
report.save(os.path.join(HOMEDIR, 'visual_inspection_report_cleaned.html'), overwrite=True)

# %%
data_mne_epochs.plot(picks="mag", n_channels=15)
# %%
filtered_data.plot(picks="mag", start = 10, duration=20, n_channels=15)
# %%
mne.chpi.compute_head_pos(raw, verbose=True)
# %%
mne.chpi.compute_chpi_amplitudes(raw, t_step_min=0.01, t_window='auto', ext_order=1, tmin=0, tmax=None, verbose=None)
# %%
mne.chpi.get_chpi_info(raw.info, on_missing='raise', verbose=None)
# %%
raw=  mne.io.read_raw_ctf(f"/users/local/Venkatesh/LSD_project/src_data/ds_data/Music/LSD/010813-4_LSD_20140903_Music.ds")
# %%
mne.preprocessing.find_bad_channels_maxwell(raw)
# %%
np.median(filtered_data.get_data())
# %%
4e-12
# %%
raw.pick_types(chpi=True)
# %%
chpi_data = mne.chpi.extract_chpi_locs(raw)

# %%
mne.chpi.extract_chpi_positions(raw, verbose=True)
# %%
raw.info['hpi_meas']
# %%
emg_channels = mne.pick_types(raw.info, meg=False, emg=True)

# %%
emg_channels
# %%
plt.plot(raw.pick('EEG061').get_data()[2400:2400])
# %%

from mne.preprocessing import ICA


ica = ICA(n_components=0.90, random_state=97, method="infomax")
ica.fit(epochs_fixed_length, picks="mag")




# %%
ica.plot_components( outlines=None)
# %%
ica.plot_sources(epochs_fixed_length,start=0, stop=20)
# %%
