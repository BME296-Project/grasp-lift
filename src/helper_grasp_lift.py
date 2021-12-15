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
    start_times = np.asarray(np.where(np.diff(truth_data[:,5])>0))
    end_times = np.asarray(np.where(np.diff(truth_data[:,5])<0))

    # Add a 2 sec "buffer" to the start each epoch 
    # Note that the buffer added is 2 seconds worth of SAMPLES, not 2 seconds
    buffered_start_times = (start_times - 2*fs).T
    
    # Calculate how long each event is
    elapsed_event_times = end_times - start_times
    # Add a buffer equal to the length of each event and 
    # This will be used in a for loop below to get rest epochs have the same
    # number of samples as the event epochs
    buffered_end_times= ((end_times + elapsed_event_times) + 2*fs).T
    
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
        
    return start_times, end_times, event_epochs, rest_epochs, epoch_duration

# %% Square Epoch

# 3-Square all values in the epoch array

def square_epoch(eeg_epochs):
    squared_epochs=eeg_epochs**2
    return squared_epochs

# %% Baseline

# 4- Within each epoch, take a window near the end to use as a baseline
def baseline_epoch(squared_epochs):
    baseline=np.zeros((34,32)) #fix me
    for index in range(len(squared_epochs)): 
        baseline[index,:]=np.mean(squared_epochs[index, 1650:2150, :], axis=0)
    return baseline

# %% Subract Baseline from Squared Entries

def subract_baseline(squared_epochs, baseline):
    subtracted_baseline=np.zeros(np.shape(squared_epochs))
    for index in range(len(squared_epochs)): 
        subtracted_baseline[index,:,:]=squared_epochs[index,:,:]-baseline[index,:]
    mean_baseline=np.mean(subtracted_baseline, axis=0)
    return subtracted_baseline, mean_baseline

# %% PLOT RESULTS 

# 7- Plot the data for electrodes C3 and C4, make qualitative observations about ERD and ERS
#   Get the Mean & StdErr across epochs of each type (motion and rest).
#   Plot the mean +/- stderr on channels you'd expect to have motor activity.

def plot_mean(mean_baseline, data, epoch_duration, channels_to_plot):
    epoch_times = np.arange(0,epoch_duration,1/data['fs'])
    channel1 = np.where(data['channels'] == channels_to_plot[0])[0][0]
    channel2 = np.where(data['channels'] == channels_to_plot[1])[0][0]
    # Mean Plot
    mean_base=np.std(mean_baseline)
    plt.plot(epoch_times,mean_baseline[:,channel1])
    plt.plot(epoch_times,mean_baseline[:,channel2])
    return epoch_times

#%% RESULTS

# Input: Predicted Events, Truth Events
# Output: Confusion Matrix

# Confusion Matrix to compare the the predicted and truth events
# We can test using the given testing data things
# Accuracy metric and ITR (becuase ITR is simple and pretty)

if __name__ == '__main__':
    print("Import Only")