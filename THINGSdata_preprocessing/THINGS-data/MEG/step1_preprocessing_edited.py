#!/usr/bin/env python3

"""
@ Lina Teichmann

    INPUTS: 
    call from command line with following inputs: 
        -participant
        -bids_dir

    OUTPUTS:
    epoched and cleaned data will be written into the preprocessing directory

    NOTES: 
    This script contains the following preprocessing steps:
    - channel exclusion (one malfunctioning channel ('MRO11-1609'), based on experimenter notes)
    - filtering (0.1 - 40Hz)
    - epoching (-100 - 1300ms) --> based on onsets of the optical sensor
    - baseline correction (zscore)
    - downsampling (200Hz)

    Use preprocessed data ("preprocessed_P{participant}-epo.fif") saved in preprocessing directory for the next steps

"""

import mne, os
import numpy as np
import pandas as pd 
from joblib import Parallel, delayed
import torch
import julius


#*****************************#
### PARAMETERS ###
#*****************************#

n_sessions                  = 12
trigger_amplitude           = 64
l_freq                      = 0.1
h_freq                      = 40
pre_stim_time               = -0.5 # -0.1
post_stim_time              = 1.0 # 1.3
std_deviations_above_below  = 4
output_resolution           = 120 #200
trigger_channel             = 'UPPT001'


#*****************************#
### HELPER FUNCTIONS ###
#*****************************#
def setup_paths(meg_dir, session):
    run_paths,event_paths = [],[]
    for file in os.listdir(f'{meg_dir}/ses-{str(session).zfill(2)}/meg/'):
        if file.endswith(".ds") and file.startswith("sub"):
            run_paths.append(os.path.join(f'{meg_dir}/ses-{str(session).zfill(2)}/meg/', file))
        if file.endswith("events.tsv") and file.startswith("sub"):
            event_paths.append(os.path.join(f'{meg_dir}/ses-{str(session).zfill(2)}/meg/', file))
    run_paths.sort()
    event_paths.sort()

    return run_paths, event_paths 

def read_raw(curr_path,session,run,participant):
    raw = mne.io.read_raw_ctf(curr_path,preload=True)
    # signal dropout in one run -- replacing values with median
    if participant == '1' and session == 11 and run == 4:  
        n_samples_exclude   = int(0.2/(1/raw.info['sfreq']))
        raw._data[:,np.argmin(np.abs(raw.times-13.4)):np.argmin(np.abs(raw.times-13.4))+n_samples_exclude] = np.repeat(np.median(raw._data,axis=1)[np.newaxis,...], n_samples_exclude, axis=0).T
    elif participant == '2' and session == 10 and run == 2: 
        n_samples_exclude = int(0.2/(1/raw.info['sfreq']))
        raw._data[:,np.argmin(np.abs(raw.times-59.8)):np.argmin(np.abs(raw.times-59.8))+n_samples_exclude] = np.repeat(np.median(raw._data,axis=1)[np.newaxis,...], n_samples_exclude, axis=0).T

    raw.drop_channels('MRO11-1609')
        
    return raw

def read_events(event_paths,run,raw):
    # load event file that has the corrected onset times (based on optical sensor and replace in the events variable)
    event_file = pd.read_csv(event_paths[run],sep='\t')
    
    # Recreate 'value' from 'things_image_nr' (e.g., 999999 for catch trials using fillna)
    if 'value' not in event_file.columns:
        if 'things_image_nr' in event_file.columns:
            event_file['value'] = pd.to_numeric(event_file['things_image_nr'], errors='coerce').fillna(999999).astype(int)
        elif 'image_nr' in event_file.columns:
            event_file['value'] = pd.to_numeric(event_file['image_nr'], errors='coerce').fillna(999999).astype(int)
        else:
            event_file['value'] = 1

    events = mne.find_events(raw, stim_channel=trigger_channel,initial_event=True)
    events = events[events[:,2]==trigger_amplitude]
    
    # BIDS events.tsv lacks 'sample'. Use optical sensor events directly, replacing the event ID with 'value'.
    # Because BIDS events are 1:1 with optical triggers, we safely assign values by length match.
    if len(events) == len(event_file):
        events[:,2] = event_file['value']
    else:
        # Fallback if optical sensor events count mismatch
        print(f"Warning: {len(events)} triggers in raw, but {len(event_file)} in events.tsv")
        events = np.zeros((len(event_file), 3), dtype=int)
        events[:, 0] = (event_file['onset'].values * raw.info['sfreq']).astype(int) + raw.first_samp
        events[:, 2] = event_file['value']
        
    return events, event_file

def concat_epochs(raw, events, event_file, session, run, epochs):
    # Map and create metadata columns as requested by the user
    if 'file_path' in event_file.columns:
        event_file = event_file.rename(columns={'file_path': 'image_path'})
        
    req_cols = ['trial_type', 'image_nr', 'category_nr', 'exemplar_nr', 'test_image_nr', 
                'things_category_nr', 'things_image_nr', 'things_exemplar_nr', 'image_path', 
                'onset', 'image_on', 'image_off', 'responded', 'key_id', 'key_time', 'RT', 
                'session_nr', 'run_nr']
                
    for col in req_cols:
        if col not in event_file.columns:
            event_file[col] = np.nan
            
    # Assign session and run numbers
    event_file['session_nr'] = session
    event_file['run_nr'] = run + 1
    
    # Filter only requested columns in order
    event_file = event_file[req_cols]

    if epochs:
        epochs_1 = mne.Epochs(raw, events, metadata=event_file, tmin = pre_stim_time, tmax = post_stim_time, picks = 'mag',baseline=None)
        epochs_1.info['dev_head_t'] = epochs.info['dev_head_t']
        epochs = mne.concatenate_epochs([epochs,epochs_1])
    else:
        epochs = mne.Epochs(raw, events, metadata=event_file, tmin = pre_stim_time, tmax = post_stim_time, picks = 'mag',baseline=None)
    return epochs

def baseline_correction(epochs):
    baselined_epochs = mne.baseline.rescale(data=epochs.get_data(),times=epochs.times,baseline=(None,0),mode='zscore',copy=False)
    epochs = mne.EpochsArray(baselined_epochs, epochs.info, epochs.events, epochs.tmin,event_id=epochs.event_id, metadata=epochs.metadata)
    return epochs

def robust_scale_and_clip(epochs, limit=20, lowq=0.25, highq=0.75):
    """
    Apply RobustScaler per channel exactly as in brainmagick (bm/norm.py),
    but working on MNE epochs data in numpy formatting, and clipping values.
    """
    data_np = epochs.get_data().copy()
    n_epochs, n_channels, n_times = data_np.shape
    
    # Flatten epochs and times to get [n_channels, n_epochs * n_times] 
    # to compute quantiles over all samples for each channel, just as in bm/norm.py
    data_2d = data_np.transpose((1, 0, 2)).reshape(n_channels, -1)
    
    scale_ = np.empty(n_channels)
    center_ = np.empty(n_channels)
    
    for d in range(n_channels):
        col = data_2d[d, :]
        quantiles = np.percentile(col, [lowq * 100, 50, highq * 100])
        low, med, high = quantiles
        scale_[d] = high - low
        center_[d] = med
        if scale_[d] == 0:
            scale_[d] = 1
            
    scale_[scale_ == 0] = 1
    
    # Scale parameters shape compatibility for [n_channels, n_epochs * n_times]
    data_2d_scaled = (data_2d - center_[:, None]) / scale_[:, None]
    
    # Clip values
    np.clip(data_2d_scaled, -limit, limit, out=data_2d_scaled)
    
    # Reshape back to [n_epochs, n_channels, n_times]
    data_scaled = data_2d_scaled.reshape(n_channels, n_epochs, n_times).transpose((1, 0, 2))
    
    # Store robust scaled and clipped data into MNE epochs structure
    epochs_scaled = mne.EpochsArray(
        data=data_scaled,
        info=epochs.info,
        events=epochs.events,
        tmin=epochs.tmin,
        event_id=epochs.event_id,
        metadata=epochs.metadata
    )
    
    return epochs_scaled

#*****************************#
### FUNCTION TO RUN PREPROCESSING ###
#*****************************#
def run_preprocessing(meg_dir,session,participant, preproc_dir, sourcedata_dir, output_resolution):
    epochs = []
    run_paths, event_paths = setup_paths(meg_dir, session)
    if not run_paths:
        return
    for run, curr_path in enumerate(run_paths):
        raw = read_raw(curr_path,session,run, participant)
        events, event_file = read_events(event_paths,run,raw)
        # raw.filter(l_freq=l_freq,h_freq=h_freq)
        epochs = concat_epochs(raw, events, event_file, session, run, epochs)
        epochs.drop_bad()
        
    print(f"Session {session} info:", epochs.info)

    # Downsample using julius BEFORE baseline correction
    current_sfreq = epochs.info['sfreq']
    if current_sfreq > output_resolution:
        data = torch.Tensor(epochs.get_data())
        old_sr = int(np.round(current_sfreq))
        resamp = julius.ResampleFrac(old_sr=old_sr, new_sr=output_resolution)
        data = resamp(data)
        
        info_kwargs = dict(epochs.info)
        info_kwargs['sfreq'] = float(output_resolution)
        info = mne.Info(**info_kwargs)
        
        epochs = mne.EpochsArray(
            data=data.numpy(),
            info=info,
            events=epochs.events,
            tmin=epochs.tmin,
            event_id=epochs.event_id,
            metadata=epochs.metadata
        )

    # Baseline correction
    epochs = baseline_correction(epochs)

    # Channelwise robust scaling
    epochs = robust_scale_and_clip(epochs, limit=20)

    # Append session identifier to ensure unique saves
    out_file = f'{preproc_dir}/preprocessed_P{str(participant)}_ses-{str(session).zfill(2)}-epo.fif'
    epochs.save(out_file, overwrite=True)
    print(f"Saved {out_file} with final sfreq={epochs.info['sfreq']}")
    
    # Explicitly clear memory
    del epochs
    return

#*****************************#
### COMMAND LINE INPUTS ###
#*****************************#
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-participant",
        required=True,
        help='participant bids ID (e.g., 1)',
    )

    parser.add_argument(
        "-bids_dir",
        required=False,
        default=None,
        help='path to bids root',
    )

    args = parser.parse_args()
    
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    import config
    
    bids_dir                    = args.bids_dir if args.bids_dir else os.path.join(config.MEG_RAW_DIR, "ds004212")
    participant                 = args.participant
    meg_dir                     = f'{bids_dir}/sub-BIGMEG{participant}/'
    sourcedata_dir              = f'{bids_dir}/sourcedata/'
    preproc_dir                 = config.MEG_PREPROCESSED_DIR
    if not os.path.exists(preproc_dir):
        os.makedirs(preproc_dir)

    ####### Run preprocessing ########
    # Run fully independently per session, downsampling and saving individually without a final stacking step to avoid OOM
    Parallel(n_jobs=4, backend="loky")(delayed(run_preprocessing)(meg_dir,session,participant,preproc_dir,sourcedata_dir,output_resolution) for session in range(1,n_sessions+1))
