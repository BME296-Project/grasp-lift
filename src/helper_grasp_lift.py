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

def epoch_data(data):

    # separate out data
    eeg_raw = data['eeg']
    truth_data = data['truth_data']
    fs = data['fs']

    # create an array of all the times the event starts (and ends)
    # np.diff takes the difference between any 2 adj data points along a given axis
    # np.where returns elements of an array that match a certain condition
    # so np.where(np.diff(rowcol_id)>0) returns all of the flash onset times 
    # from the col corresponding to "release"
    start_times = np.asarray(np.where(np.diff(truth_data[:,5])>0))
    end_times = np.asarray(np.where(np.diff(truth_data[:,5])<0))

    # Add a 1 sec "buffer" to the start and end times
    start_times = (start_times - fs).T
    end_times = (end_times + fs).T
    
    # Get epoch parameters
    samples_per_epoch = (end_times - start_times)[0][0] # number of samples per epoch
    epoch_duration = int(samples_per_epoch/fs)          # length of each epoch in seconds
    num_epochs = int(len(start_times))
    num_channels = eeg_raw.shape[1]

    # NOTE: Add in Frequency and length of the eeg data things
    # pages x rows x cols
    eeg_epochs = np.zeros([num_epochs, samples_per_epoch, num_channels])

    # for each page, 
    for sample_index in range(len(start_times)):
        eeg_epochs[sample_index,:,:] = eeg_raw[start_times[sample_index][0]:end_times[sample_index][0],:]

    return start_times, end_times, eeg_epochs

# %% NEXT STEPS

# 3-Square all values in the epoch array

# 4- Within each epoch, take a window near the end to use as a baseline

# 5- Meet with Jangraw again when he is not late for another meeting to geed feedback and go over next steps which to the best of my knowledge are...

# 6- Subtract baseline from squared entries of each epoch to normalize data

# %% PLOT RESULTS 

# 7- Plot the data for electrodes C3 and C4, make qualitative observations about ERD and ERS

# 8- Write report as if this was our original analysis all along and pretend we didn't change it halfway through


#%% RESULTS

# Input: Predicted Events, Truth Events
# Output: Confusion Matrix

# Confusion Matrix to compare the the predicted and truth events
# We can test using the given testing data things
# Accuracy metric and ITR (becuase ITR is simple and pretty)

if __name__ == '__main__':
    print("Import Only")