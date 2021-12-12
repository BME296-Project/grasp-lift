#!/usr/bin/env python3

"""
helper_grasp_lift.py

functions to help with data analysis. this includes epoching
data, making classifications, predictors and plotting results

@author: george
"""

# %% IMPORTS
import numpy as np
from matplotlib import pyplot as plt

# %% EPOCH THE DATA

# Input: Raw EEG Data, start time, end time
# Output: 3d array of epoched events. Raw data turned into chunks of data
# NOTE: Look at experiement to see how it was set up. There may have been rough timing for each events. Then we can epoch around the center of that time window? 
# NOTE: Check data for start and end point of potnetial epochs -> compare across subject

def epoch_data(data, epoch_duration=1):

    # separate out data
    eeg_raw = data['eeg']
    truth_data = data['truth_data']
    fs = data['fs']

    # change different epoch sizes, default to be 1 second epoch pages
    samples_per_epoch = int(fs*epoch_duration)
    num_trials = int(len(eeg_raw)/samples_per_epoch)
    num_channels = eeg_raw.shape[1]
    num_events = truth_data.shape[1]

    # NOTE: Add in Frequency and length of the eeg data things
    # pages x rows x cols
    eeg_epoch = np.zeros([num_trials, samples_per_epoch, num_channels])
    truth_epoch = np.zeros([num_trials, samples_per_epoch, num_events])

    # for each page, 
    for sample_index in range(num_trials):
        trial_index = sample_index * samples_per_epoch
        eeg_epoch[sample_index,:,:] = eeg_raw[trial_index:trial_index+samples_per_epoch,:]
        truth_epoch[sample_index,:,:] = truth_data[trial_index:trial_index+samples_per_epoch,:]
        

    return eeg_epoch, truth_epoch


#%% TAKE FFT

# Filter Data using fft things -> hilbert tranforms? scipy.filter.filtfilt()?
# Input: Raw EEG data
# Output: Filtered EEG Data
# take FFT
# Take mean power in 5-35Hz range (beta + mu rhythms)

def take_FFT(eeg_epoch, fs):
    # Take FFT of signal 
    signal_fft = np.fft.rfft(eeg_epoch)
    
    # Get frequencies of FFT
    signal_fft_freqs = np.fft.rfftfreq(len(eeg_epoch)) * fs
    
    # create band-stop filter to remove all signals outside of 5-35hz
    filter_frequncy_response = np.ones(len(signal_fft_freqs))
    filter_frequncy_response[signal_fft_freqs<=5] = 0 
    filter_frequncy_response[signal_fft_freqs>=35] = 0 
    
    # Multiply signal FFT by frequncy response
    filtered_fft = signal_fft * filter_frequncy_response
    
    # # Take inverse FFT of filtered signal
    # filtered_signal = np.fft.irfft(filtered_fft)

    # Get spectrum (signal power)
    signal_power = filtered_fft *filtered_fft.conj()
    # normalize spectrum
    signal_power_norm = signal_power/np.max(signal_power)
    # convert to decibels
    signal_power_db = 10*np.log10(signal_power_norm)
    
    

    # # Plot the power spectrum in dB
    # plt.figure(453)
    # plt.clf()
    # plt.plot(signal_fft_freqs, signal_power_db)
    
    # # Annotate the plot
    # plt.title('signal power spectrum')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Power (dB)')
    # plt.tight_layout()
    
    # mean_power = np.mean(signal_power_db)
    
    
    
    return filtered_fft, signal_power_db

#%% CLASSIFICATIONS

# Use an SVM to find the best weighted combination across electrodes
# Generate or predict threshold values for the data. 
# Input: Epoched EEG Data, filtered, start event times
# Output: Predicted Events

def train_classification(eeg_epoch, truth_epoch, data_channels, channels):

    # label each eeg epoch using truth data
   

    # get the number of trials we have
    num_trials = eeg_epoch.shape[0]

    # initialize arrays
    mean_eeg = np.zeros([num_trials]) # eeg means
    mean_truth = np.zeros([num_trials, 6]) # truth means

    # get indexs for channels
    channel_index = []
    for channel in channels:
        index = np.where(data_channels==channel)[0][0] # assumes 'channel' is valid 
        channel_index.append(index)

    # for each of the trials, compare to truth data
    for trial_index in range(num_trials):
        mean_eeg = np.mean(eeg_epoch[trial_index, :, channel_index])
        mean_truth[trial_index,:] = np.mean(truth_epoch[trial_index, :, :], 0) 


# Taking in the filtered data and some channels (realistically we should focus on just one for now) that we want to look at
# Then generate a prediction based on a threshold of some sort

# Some variable names for predicted events
# is_predicted_hand_start
# is_predicted_first_touch
# is_predicted_grasp
# is_predicted_lift
# is_predicted_replace
# is_predicted_release

#%% RESULTS

# Input: Predicted Events, Truth Events
# Output: Confusion Matrix

# Confusion Matrix to compare the the predicted and truth events
# We can test using the given testing data things
# Accuracy metric and ITR (becuase ITR is simple and pretty)

if __name__ == '__main__':
    print("Import Only")