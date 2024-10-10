import os
import argparse
import mne
from autoreject import AutoReject
from mne.preprocessing import ICA
from mne.report import Report
from joblib import Parallel, delayed
from multiprocessing import Lock

# Hardcoded BIDS and derivatives directories
BIDS_DIR = '/users/local/Venkatesh/LSD_project/src_data/fif_data_BIDS/LSD/'
DERIVATIVES_DIR = '/users/local/Venkatesh/LSD_project/src_data/derivatives/LSD/'

# Define the function to preprocess EEG data
def preprocess_meg(subject_id, input_dir, power_line_freq=50):
    # Construct the path to the subject's data in BIDS format
    input_file = os.path.join(input_dir, f'sub-{subject_id}', f'ses-01', 'meg', f'sub-{subject_id}_ses-01_task-music_meg.fif')

    report = Report(title='MEG Report for Subject {}'.format(subject_id))

    # Step 1: Power line artifact removal using notch filter
    raw = mne.io.read_raw_fif(input_file, preload=True)
    report.add_raw(raw, title=f'Subject {subject_id} - Pre powerline artefact removal', psd=True)
    
    raw.notch_filter(freqs=[power_line_freq, 2*power_line_freq, 3*power_line_freq, 4*power_line_freq, 5*power_line_freq])
    report.add_raw(raw, title=f'Subject {subject_id} - Power Line Artifact Removal', psd=True)

    # Step 2: Low-pass and High-pass filtering
    raw.filter(l_freq=1., h_freq=250.)  # High-pass at 1 Hz, Low-pass at 40 Hz
    report.add_raw(raw, title=f'Subject {subject_id} - Bandpass Filtering', psd=True)

    # Step 3: ICA works best on the data without bad epochs. https://autoreject.github.io/stable/auto_examples/plot_autoreject_workflow.html
    # Thus, first apply AutoReject to remove bad epochs
    
    raw_for_meg_picks = raw.copy().pick_types(meg=True, eeg=True, eog=False) # just using `raw` results in error by AutoReject while doing Picks. This gets resolved by specifying picks
    epochs = mne.make_fixed_length_epochs(raw_for_meg_picks, duration=2, preload=True, proj=False)
    ar = AutoReject(picks="mag", random_state=99)
    ar.fit(epochs)
    epochs_clean, reject_log = ar.transform(epochs, return_log=True)
    report.add_figure(reject_log.plot('horizontal'), title=f'Subject {subject_id} - Autoreject Log')
    report.add_epochs(epochs_clean, title=f'Subject {subject_id} - Autoreject Applied for Epochs')

    # Step 3:ICA 
    # Second, apply ICA to remove artifacts
    ica = ICA(n_components=20, random_state=97, method="picard")
    ica.fit(epochs[~reject_log.bad_epochs], picks='mag')
    
    # # step 3.1 ECG artifact removal
    ica.exclude = []
    eog_indices, eog_scores = ica.find_bads_eog(epochs, ch_name=["EEG057", "EEG058"])
    ecg_indices, ecg_scores = ica.find_bads_ecg(epochs, ch_name="EEG059")
    report.add_ica(ica, inst=None, title=f'Subject {subject_id}, IC; EOG match with EOG channels: {eog_indices}',  eog_scores=eog_scores)
    report.add_ica(ica, inst=None, title=f'Subject {subject_id}, IC; ECG match with ECG channels: {ecg_indices}',  ecg_scores=ecg_scores)
    # ica.exclude = ecg_indices +
    
    # ica.exclude.extend(eog_indices)
    # ica.apply(raw)
    # report.add_raw(raw, title=f'Subject {subject_id} - Muscle Artifacts Removal using ICA', psd=True)

    # # Save the cleaned data to a new FIF file
    output_file = os.path.join(DERIVATIVES_DIR, f'sub-{subject_id}', 'meg', f'sub-{subject_id}_eeg_cleaned.fif')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    raw.save(output_file, overwrite=True)
    
    report.save(os.path.join(DERIVATIVES_DIR, f'sub-{subject_id}', f'sub-{subject_id}_report.html'), overwrite=True)


# Main script to run preprocessing per subject
def main():
    parser = argparse.ArgumentParser(description='MEG Preprocessing Script with MNE and AutoReject')
    parser.add_argument('-s', '--subjects', type=str, required=True, nargs='+', help='List of subject IDs to preprocess')
    
    args = parser.parse_args()
    eog_indices = dict()
    # eog_indices['01'] = []
    # Create a single report for all subjects
        
    Parallel(n_jobs=-1)(delayed(preprocess_meg)(subject_id, BIDS_DIR) for subject_id in args.subjects)

    # Save the combined report
    # output_html = os.path.join(DERIVATIVES_DIR, 'meg_preprocessing_report.html')
    # report.save(output_html, overwrite=True)


if __name__ == '__main__':
    main()
    

#%%

# #%%
# for i in range(11):
#     idx = f'{i+1:02}'
#     data_mne = mne.io.read_raw_fif(f'/users/local/Venkatesh/LSD_project/src_data/fif_data_BIDS/PLA/sub-{idx}/ses-01/meg/sub-{idx}_ses-01_task-music_meg.fif')

#     plt.plot(data_mne.get_data(picks='eeg')[2, 1200:4000].T)#, label=["EEG057", "EEG058", "EEG059", "EEG061", "EEG062", "EEG063", "EEG064"])
#     plt.title(idx)
#     plt.show()
# # # %%

# plt.plot(data_mne.get_data(picks='eeg')[2, 1200:12000].T, label=["EEG057", "EEG058", "EEG059", "EEG061", "EEG062", "EEG063", "EEG064"])
# # %%
# plt.plot(data_mne.get_data(picks='eeg')[:5, 1200:5*1200].T, label=["EEG057", "EEG058", "EEG059", "EEG061", "EEG062"])
# plt.legend()
# %%
# import mne
# import matplotlib.pyplot as plt

# for i in range(11):
#     idx = f'{i+1:02}'
#     if i == 3:
#         continue
#     data_mne = mne.io.read_raw_fif(f'/users/local/Venkatesh/LSD_project/src_data/fif_data_BIDS/PLA/sub-{idx}/ses-01/meg/sub-{idx}_ses-01_task-music_meg.fif')
#     plt.plot(data_mne.get_data(picks='eeg')[2, 1200:5*1200].T)
#     # plt.title(f"EEG0{i+57}")
#     plt.grid()
#     plt.show()

# %%
# # #"ecg": 059
# plt.plot(data_mne.get_data(picks='eeg')[0, 1200:12000].T-data_mne.get_data(picks='eeg')[1, 1200:12000].T)
# # # %%

# # %%

# %%
