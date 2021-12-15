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
from scipy import fft
from scipy.signal import firwin, freqz, filtfilt, hilbert

# %% MAKING A BAND PASS FILTER
# Make band pass filter to get rid off all data not in mu rhythm range (8-13Hz)

def filter_data(data, low_cutoff=8, high_cutoff=13, filter_order=5, fs=500, filter_type='hann'):
    '''
    This function makes a filter based on the parameters passed in and returns
    the coefficients for the filter, then filters a signal using the 
    filter coefficients.


    Parameters
    ----------
    data : dictionary
        raw data and realted channels and data information.
    low_cutoff : int
        lower cufoff frequency in Hz.
    high_cutoff : int
        higher cutoff frequency in Hz.
    filter_order : int
        The order of the filter to be made
    fs : int
        Sampling frequency of the data.
    filter_type : string, optional
        type of filter. defines how the filter performs. The default is "hann".

    Returns
    -------
    filtered_data : np 2d array
        the filtered data through the filter. size num_channels x num_samples

    '''
    
    # Create an empty array for filter_coefficients
    filter_coefficients = np.zeros(filter_order)
    
    # Create the FIR filter using scipy.firwin
    numtaps = filter_order + 1
    filter_coefficients = firwin(numtaps, cutoff=[low_cutoff, high_cutoff], fs=fs, window=filter_type, pass_zero='bandpass')
    
    # Now filter the raw eeg data
    # Extract the raw data from the data dictionary
    raw_data = data['eeg']
    
    # Use scipy.filter.filtfilt to apply the filter forward/backward in time 
    # to each channel in the raw data    
    filtered_data= filtfilt(b=filter_coefficients, a=1, x=raw_data)
    
    return filtered_data

# %% EPOCH THE DATA

# Input: Raw EEG Data, start time, end time
# Output: 3d array of epoched events. Raw data turned into chunks of data
# NOTE: Look at experiement to see how it was set up. There may have been rough timing for each events. Then we can epoch around the center of that time window? 
# NOTE: Check data for start and end point of potnetial epochs -> compare across subject

def epoch_data(data, filtered_data):

    # separate out data
    truth_data = data['truth_data']
    fs = data['fs']

    # create an array of all the times the event starts (and ends)
    # np.diff takes the difference between any 2 adj data points along a given axis
    # np.where returns elements of an array that match a certain condition
    # so np.where(np.diff(rowcol_id)>0) returns all of the flash onset times 
    # from the col corresponding to "release"
    start_times = np.asarray(np.where(np.diff(truth_data[:,5])>0)).T
    end_times = np.asarray(np.where(np.diff(truth_data[:,5])<0)).T

    # Add a 2 sec "buffer" to the start each epoch 
    # Note that the buffer added is 2 seconds worth of SAMPLES, not 2 seconds
    buffered_start_times = (start_times - 2*fs)
    
    # Calculate how long each event is
    elapsed_event_times = end_times - start_times
    # This will be used in a for loop below to get rest epochs have the same
    # number of samples as the event epochs
    # Now add a buffer equal to the length of each event +2 secs
    # This makes it so that all epochs are the same size
    buffered_end_times= ((end_times + elapsed_event_times) + 2*fs)
    
    # Get epoch parameters
    samples_per_epoch = (end_times - buffered_start_times)[0][0] # number of samples per epoch
    epoch_duration = float(samples_per_epoch/fs)          # length of each epoch in seconds
    num_epochs = int(len(start_times))
    num_channels = filtered_data.shape[1]

    # NOTE: Add in Frequency and length of the eeg data things
    # pages x rows x cols
    event_epochs = np.zeros([num_epochs, samples_per_epoch, num_channels])
    rest_epochs = np.zeros([num_epochs, samples_per_epoch, num_channels])

    # for each page, 
    for sample_index in range(len(start_times)):
        event_epochs[sample_index,:,:] = filtered_data[buffered_start_times[sample_index][0]:end_times[sample_index][0],:]
        rest_epochs[sample_index,:,:] = filtered_data[end_times[sample_index][0]:buffered_end_times[sample_index][0],:]
        
    return start_times, end_times, buffered_start_times, buffered_end_times, event_epochs, rest_epochs, epoch_duration

# %% Square Epoch

# 3-Square all values in the epoch array

def square_epoch(event_epochs, rest_epochs):
    squared_event_epochs=event_epochs**2
    squared_rest_epochs=rest_epochs**2
    return squared_event_epochs, squared_rest_epochs

# %% Baseline

# 4- Within each epoch, take a window near the end to use as a baseline
def get_baselines(data, squared_rest_epochs):
    fs = data['fs']
    
    # Get indexes representing last 1 sec worth of rest data in an epoch
    # where A is the start of that last second and B is the end
    B = np.shape(squared_rest_epochs)[1]
    A = B-fs
    
    baselines=np.zeros((34,32)) #fix me
    for r_index in range(len(squared_rest_epochs)): 
        baselines[r_index,:]=np.mean(squared_rest_epochs[r_index, A:B, :], axis=0)
    return baselines

# %% Subract Baseline from Squared Entries

def subract_baseline(squared_event_epochs, squared_rest_epochs, baselines):
    events_minus_baseline=np.zeros(np.shape(squared_event_epochs))
    for index in range(len(squared_event_epochs)): 
        events_minus_baseline[index,:,:]=squared_event_epochs[index,:,:]-baselines[index,:]
    
    rests_minus_baseline=np.zeros(np.shape(squared_rest_epochs))
    for index in range(len(squared_rest_epochs)): 
        rests_minus_baseline[index,:,:]=squared_rest_epochs[index,:,:]-baselines[index,:]
    
    return events_minus_baseline, rests_minus_baseline

# %% GET MEAN AND STANDARD ERROR

def get_mean_SE(events_minus_baseline, rests_minus_baseline):
    # Get the mean signal across all epochs for each channel
    mean_events=np.mean(events_minus_baseline, axis=0)
    mean_rests=np.mean(rests_minus_baseline, axis=0)
    # Get the standard deviation of the signal across all epochs for each channel
    events_std = np.std(events_minus_baseline, axis=0)
    rests_std = np.std(rests_minus_baseline, axis=0)
    # Get the standard error of the the signal across all epochs for each channel
    events_se = events_std/np.sqrt(np.shape(events_minus_baseline)[1])
    rests_se = rests_std/np.sqrt(np.shape(rests_minus_baseline)[1])

    return mean_events, mean_rests, events_se, rests_se
# %% PLOT RESULTS 

# 7- Plot the data for electrodes C3 and C4, make qualitative observations about ERD and ERS
#   Get the Mean & StdErr across epochs of each type (motion and rest).
#   Plot the mean +/- stderr on channels you'd expect to have motor activity.

# Plot the mean +/- stderr on channels you'd expect to have motor activity. 
# Are motion and rest epochs separated at the times and locations you'd expect?

def plot_results(mean_events, mean_rests, events_se, rests_se, data, epoch_duration, channels_to_plot):
    epoch_times = np.arange(0,epoch_duration,1/data['fs'])
    channel1 = np.where(data['channels'] == channels_to_plot[0])[0][0]
    channel2 = np.where(data['channels'] == channels_to_plot[1])[0][0]
    # Plot event means for channels 1 and 2
    channel1_means = plt.plot(epoch_times, mean_events[:, channel1], label='Channel 1 means')
    channel2_means = plt.plot(epoch_times, mean_events[:, channel2], label='Channel 2 means')
    
    # Get upper limit for channel 1
    channel1_UL = mean_events[:, channel1] + events_se[:, channel1]
    # Get lower limit for channel 1
    channel1_LL = mean_events[:, channel1] - events_se[:, channel1]
    # Get upper limit for channel 2
    channel2_UL = mean_events[:, channel2] + events_se[:, channel2]
    # Get lower limit for channel 2
    channel2_LL = mean_events[:, channel2] - events_se[:, channel2]
    
    # plot CI for event means using mean +/- stderr for channels 1 and 2
    plt.fill_between(epoch_times, channel1_UL, channel1_LL, alpha=0.5, lw=5, label='Channel 1 +/- Standard Error')
    plt.fill_between(epoch_times, channel2_UL, channel2_LL, alpha=0.5, lw=5, label='Channel 2 +/- Standard Error')
    

