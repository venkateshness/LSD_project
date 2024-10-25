import os
import argparse
import mne
from autoreject import AutoReject
from mne.preprocessing import ICA
from mne.report import Report
from joblib import Parallel, delayed

def preprocess_meg(subject_id, input_dir, task, drug, eog_components, DERIVATIVES_DIR, power_line_freq=50):
    input_file = os.path.join(input_dir, f'sub-{subject_id}', f'ses-01', 'meg', f'sub-{subject_id}_ses-01_task-{task}_meg.fif')
    
    os.makedirs(os.path.join(DERIVATIVES_DIR, f'sub-{subject_id}'), exist_ok=True)
    os.makedirs(os.path.join(DERIVATIVES_DIR, f'sub-{subject_id}', 'meg'), exist_ok=True)
    
    report = Report(title=f'MEG Report for Subject {subject_id}', verbose=False)
    
    # Step 1: Power line artifact removal using notch filter
    raw = mne.io.read_raw_fif(input_file, preload=True, verbose=False)
    report.add_raw(raw, title=f'Subject {subject_id} - Pre powerline artefact removal', psd=True)
    
    raw.notch_filter(freqs=[power_line_freq, 2*power_line_freq, 3*power_line_freq, 4*power_line_freq, 5*power_line_freq], verbose=False)
    report.add_raw(raw, title=f'Subject {subject_id} - Power Line Artifact Removal', psd=True)
    raw.save(os.path.join(DERIVATIVES_DIR, f'sub-{subject_id}', 'meg', f'sub-{subject_id}_raw_notch_meg.fif'), overwrite=True)

    # Step 2: Low-pass and High-pass filtering
    raw.filter(l_freq=1, h_freq=250, verbose=False)  # High-pass at 1 Hz, Low-pass at 40 Hz
    report.add_raw(raw, title=f'Subject {subject_id} - Bandpass Filtering', psd=True)
    raw.save(os.path.join(DERIVATIVES_DIR, f'sub-{subject_id}', 'meg', f'sub-{subject_id}_raw_filtered_meg.fif'), overwrite=True)

    # Step 3: ICA works best on the data without bad epochs. https://autoreject.github.io/stable/auto_examples/plot_autoreject_workflow.html
    # Thus, first apply AutoReject to remove bad epochs
    events = mne.make_fixed_length_events(raw, duration=2, overlap=0)
    epochs = mne.Epochs(raw, events, event_id=1, tmin=0, tmax=1.999, baseline=None, preload=True, picks='meg')
    ar = AutoReject(picks="mag",n_jobs=-1, random_state=99, n_interpolate=[1, 4, 8, 16, 32])
    ar.fit(epochs)
    
    reject_log = ar.get_reject_log(epochs, picks='mag')
    report.add_figure(reject_log.plot('horizontal'), title=f'Subject {subject_id} - Autoreject Log')
    report.add_epochs(epochs, title=f'Subject {subject_id} - Autoreject Applied for Epochs')
    # epochs.save(os.path.join(DERIVATIVES_DIR, f'sub-{subject_id}', 'meg', f'sub-{subject_id}_AR_PreICA_ft_epochs_meg.fif'), overwrite=True)
    
    output_file_reject_log = os.path.join(DERIVATIVES_DIR, f'sub-{subject_id}', 'meg', f'sub-{subject_id}_reject_log_AR_pre_meg.fif')
    reject_log.save(output_file_reject_log, overwrite=True)

    # Step 3:ICA 
    # Second, apply ICA to remove artifacts
    ica = ICA(n_components=20, random_state=97, method="picard")
    ica.fit(epochs[~reject_log.bad_epochs], picks="mag")
    ica.save(os.path.join(DERIVATIVES_DIR, f'sub-{subject_id}', 'meg', f'sub-{subject_id}_ica_meg.fif'), overwrite=True)
    
    # step 3.1 ECG / EOG artifact removal
    # ica.exclude = []
    ecg_epochs = mne.preprocessing.create_ecg_epochs(raw, ch_name="EEG059")
    ecg_indices, ecg_scores = ica.find_bads_ecg(ecg_epochs, ch_name="EEG059")

    eog_epochs = mne.preprocessing.create_eog_epochs(raw, ch_name=["EEG057", "EEG058"])
    eog_indices, eog_scores = ica.find_bads_eog(eog_epochs, ch_name=["EEG057", "EEG058"])
        
    report.add_ica(ica, inst=None, title=f'Subject {subject_id}, IC; EOG match with EOG channels: {eog_indices}',  eog_scores=eog_scores,  n_jobs=-1)
    report.add_ica(ica, inst=None, title=f'Subject {subject_id}, IC; ECG match with ECG channels: {ecg_indices}',  ecg_scores=ecg_scores,  n_jobs=-1)

    ## ICA on epochs_clean
    ecg_components = ecg_indices
    eog_components = eog_components[task][drug][f"{subject_id}"]
    to_exclude = list(set(ecg_components + eog_components))
    ica.exclude = to_exclude
    epochs_ar_clean_ICA = ica.apply(epochs, exclude=to_exclude)
    
    report.add_ica(ica, inst=None, title=f'Subject {subject_id} - ICA Applied for Epochs',  n_jobs=-1)
    report.add_epochs(epochs_ar_clean_ICA, title=f'Subject {subject_id} - ICA Applied for Epochs')
    
    # Step 4: Apply AutoReject to remove bad epochs post-ICA
    ar = AutoReject(picks="mag",n_jobs=-1, random_state=99, n_interpolate=[1, 4, 8, 16, 32])
    ar.fit(epochs_ar_clean_ICA)
    epochs_ar_clean_ICA_ar, reject_log_post_ICA = ar.transform(epochs_ar_clean_ICA, return_log=True)
    report.add_figure(reject_log_post_ICA.plot('horizontal'), title=f'Subject {subject_id} - Autoreject Log post_ICA')

    # # Save the cleaned data (Epochs) to a new FIF file
    output_file = os.path.join(DERIVATIVES_DIR, f'sub-{subject_id}', 'meg', f'sub-{subject_id}_cleaned_epochs_meg.fif')
    epochs_ar_clean_ICA_ar.save(output_file, overwrite=True)

    report.add_epochs(epochs_ar_clean_ICA_ar, title=f'Subject {subject_id} - Cleaned Epochs', psd=True)

    report.save(os.path.join(DERIVATIVES_DIR, f'sub-{subject_id}', f'sub-{subject_id}_report.html'), overwrite=True)
    report.save(os.path.join(f"/users/local/Venkatesh/LSD_project/Preprocessing_Reports/{task}/{drug}", f'sub-{subject_id}_report.html'), overwrite=True)

    output_file_reject_log = os.path.join(DERIVATIVES_DIR, f'sub-{subject_id}', 'meg', f'sub-{subject_id}_reject_log_AR_post_meg.fif')
    reject_log_post_ICA.save(output_file_reject_log, overwrite=True)

def main():
    
    parser = argparse.ArgumentParser(description='MEG Preprocessing Script with MNE and AutoReject')
    parser.add_argument('-s', '--subjects', type=str, required=True, nargs='+', help='List of subject IDs to preprocess')
    parser.add_argument('-t', '--task', type=str, required=True, help='Task name (e.g., music, video)')
    parser.add_argument('-d', '--drug', type=str, required=True, help='Drug condition (e.g., LSD, PLA)')
    
    
    args = parser.parse_args()
    
    ###############################################
    #                                             #
    #           EOG components handpicked         #
    #                                             #
    ###############################################

    eog_components = {

    "Music": {

    "LSD": {

    "01": [7, 11],

    "02": [0, 2, 4, 5],

    "03": [10, 13],

    "04": [0, 1],

    "05": [12, 15],

    "06": [17],

    "07": [10, 15],

    "08": [1, 3],

    "09": [0, 14],

    "10": [0],

    "11": [3]

    },

    "PLA": {

    "01": [8],

    "02": [19],

    "03": [9],

    "04": [14],

    "05": [18, 19],

    "06": [17, 18],

    "07": [],  

    "08": [0, 12, 17],

    "09": [9, 10, 14],

    "10": [13],

    "11": [0, 1, 3, 9]

    }

    },

    "Video": {

    "LSD": {

    "01": [7, 9],

    "02": [5, 11],

    "03": [0, 3, 6],

    "04": [2, 3, 4],

    "05": [0, 2],

    "06": [2, 5],

    "07": [0, 1],

    "08": [8, 13],

    "09": [0, 1],

    "10": [0, 1],

    "11": [0, 1]

    },

    "PLA": {

    "01": [3, 11],

    "02": [0, 3],

    "03": [0, 1, 2, 8],

    "04": [0, 12],

    "05": [0, 13],

    "06": [0, 9],

    "07": [15, 16],

    "08": [0, 16],

    "09": [1, 3],

    "10": [0, 1],

    "11": [0, 1, 2]

    }

    }

    }

################################################################################
    
    # Hardcoded BIDS and derivatives directories
    BIDS_DIR = f'/users/local/Venkatesh/LSD_project/src_data/fif_data_BIDS/{args.task}/{args.drug}/'
    DERIVATIVES_DIR = f'/users/local/Venkatesh/LSD_project/src_data/derivatives/{args.task}/{args.drug}/'
    
    task = args.task
    drug = args.drug
    

    Parallel(n_jobs=-1)(delayed(preprocess_meg)(subject_id, BIDS_DIR, task, drug, eog_components, DERIVATIVES_DIR) for subject_id in args.subjects)

    # Save the combined report
    # output_html = os.path.join(DERIVATIVES_DIR, 'meg_preprocessing_report.html')
    # report.save(output_html, overwrite=True)


if __name__ == '__main__':
    main()
    

#%%

# #%%
# for i in range(11):
#     idx = f'{i+1:02}'

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
# import mne
# data_mne = mne.io.Raw("/users/local/Venkatesh/LSD_project/src_data/fif_data_BIDS/Music/LSD/sub-01/ses-01/meg/sub-01_ses-01_task-Music_meg.fif", preload=True)
# # data_mne
# annot, score= mne.preprocessing.annotate_muscle_zscore(data_mne,ch_type="mag",
#     threshold=4,
#     min_length_good=0.2,
#     filter_freq=[110, 140])
# # # %%
# # annot
# # # %%
# ###############################################
# #                                             #
# #                 EOG Data Code               #
# #                                             #
# ###############################################

# # %%
# plt.plot(data_mne.times, score)
# # %%
# len(np.where(score>3)[0])
# %%
