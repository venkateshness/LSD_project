import os
import argparse
import mne
from autoreject import AutoReject
from mne.preprocessing import ICA
from mne.report import Report
from joblib import Parallel, delayed
import numpy as np

def preprocess_meg(subject_id, input_dir, task, drug, DERIVATIVES_DIR, power_line_freq=50):
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
    raw.filter(l_freq=1, h_freq=125, verbose=False)  # High-pass at 1 Hz, Low-pass at 40 Hz
    report.add_raw(raw, title=f'Subject {subject_id} - Bandpass Filtering', psd=True)
    raw.save(os.path.join(DERIVATIVES_DIR, f'sub-{subject_id}', 'meg', f'sub-{subject_id}_raw_filtered_meg.fif'), overwrite=True)

    # Step 3: ICA works best on the data without bad epochs. https://autoreject.github.io/stable/auto_examples/plot_autoreject_workflow.html
    # Thus, first apply AutoReject to remove bad epochs
    events = mne.make_fixed_length_events(raw, duration=2, overlap=0)
    epochs = mne.Epochs(raw, events, event_id=1, tmin=0, tmax=1.999, baseline=None, preload=True, picks='meg')
    
    # step 3.1: Fetch from fif_data_BIDS, where it's manually annotated, the bad epochs
    #######Rremoved for RestC.. first run
    # manually_annotated_good_epochs = np.load(f'/users/local/Venkatesh/LSD_project/src_data/fif_data_BIDS_epochs/{task}/{drug}/sub-{subject_id}/meg/sub-{subject_id}_good_epochs_upon_visual_inspection_of_raw_filtered_epochs.npz')['arr_0']
    # epochs = epochs[manually_annotated_good_epochs]
    # epochs.save(os.path.join(DERIVATIVES_DIR, f'sub-{subject_id}', 'meg', f'sub-{subject_id}_epochs_meg.fif'), overwrite=True)
    
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
    eog_components = eog_indices
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

    # eog_components = {

    # "Music": {

    # "LSD": {

    # "010": [8, 11],

    # "015": [0, 1, 2, 4, 5, 12],

    # "016": [7, 8, 3, 5, 13], 

    # "013": [0, 1, 7],

    # "006": [19],

    # "005": [2],

    # "011": [14, 19],

    # "003": [0, 1, 2],

    # "018": [0, 11],

    # "017": [0, 18, 19, 4],

    # "009": [0, 1, 3, 4, 5, 11]

    # },

    # "PLA": {

    # "011": [8],

    # "010": [19],

    # "005": [9],

    # "017": [15],

    # "018": [19],

    # "003": [17],

    # "006": [],

    # "009": [8, 16],

    # "013": [8, 14, 11],

    # "016": [14],

    # "015": [0, 1, 3, 6]

    # }

    # },

    # "Video": {

    # "LSD": {

    # "011": [11,  17],

    # "016": [0, 1, 2, 3, 4, 5, 6, 7, 10 ],

    # "006": [0, 3, 6, 15],

    # "015": [1, 2, 3, 4, 11],

    # "003": [0, 2],

    # "010": [2, 5],

    # "013": [0, 1],

    # "005": [8, 13],

    # "018": [0, 1, 12],

    # "017": [0, 1, 7, 8, 9, 19],

    # "009": [0, 1, 10]

    # },

    # "PLA": {

    # "011": [4],

    # "018": [0, 2],

    # "015": [0, 1, 2, 5],

    # "010": [0, 13],

    # "017": [0, 13, 19],

    # "016": [0, 9, 19],

    # "005": [15, 16],

    # "006": [0, 18],

    # "003": [1, 3],

    # "009": [0, 1],

    # "013": [0, 1]

    # }

    # }

    # }

################################################################################
    
    # Hardcoded BIDS and derivatives directories
    BIDS_DIR = f'/users/local/Venkatesh/LSD_project/src_data/fif_data_BIDS/{args.task}/{args.drug}/'
    DERIVATIVES_DIR = f'/users/local/Venkatesh/LSD_project/src_data/derivatives/func/{args.task}/{args.drug}/'
    
    task = args.task
    drug = args.drug
    

    Parallel(n_jobs=-1)(delayed(preprocess_meg)(subject_id, BIDS_DIR, task, drug, DERIVATIVES_DIR) for subject_id in args.subjects)

    # Save the combined report
    # output_html = os.path.join(DERIVATIVES_DIR, 'meg_preprocessing_report.html')
    # report.save(output_html, overwrite=True)


if __name__ == '__main__':
    main()
    
