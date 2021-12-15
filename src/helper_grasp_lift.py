#!/usr/bin/env python3

"""
helper_grasp_lift.py

functions to help with data analysis. this includes epoching
filter data, create epochs, square and get baseline for data, 
get the mean and standard error of the data, plot results.

@author: anna jane, brendan, george, jason

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
    
    # return data
    return filtered_data

# %% EPOCH THE DATA

def epoch_data(data, filtered_data):
    '''
    

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    filtered_data : TYPE
        DESCRIPTION.

    Returns
    -------
    start_times : TYPE
        DESCRIPTION.
    end_times : TYPE
        DESCRIPTION.
    buffered_start_times : TYPE
        DESCRIPTION.
    buffered_end_times : TYPE
        DESCRIPTION.
    event_epochs : TYPE
        DESCRIPTION.
    rest_epochs : TYPE
        DESCRIPTION.
    epoch_duration : TYPE
        DESCRIPTION.

    '''

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

    # for each page, get the data for that epoch and save to new array
    for sample_index in range(len(start_times)):
        event_epochs[sample_index,:,:] = filtered_data[buffered_start_times[sample_index][0]:end_times[sample_index][0],:]
        rest_epochs[sample_index,:,:] = filtered_data[end_times[sample_index][0]:buffered_end_times[sample_index][0],:]
        
    # Convert epochs from SAMPLES to SECONDS
    event_epochs = event_epochs / fs
    rest_epochs = rest_epochs / fs
    
    return start_times, end_times, buffered_start_times, buffered_end_times, event_epochs, rest_epochs, epoch_duration

# %% Square Epoch

def square_epoch(event_epochs, rest_epochs):
    '''
    square all the values in the epoch. This is close to how power calculation happens

    Parameters
    ----------
    event_epochs : TYPE
        DESCRIPTION.
    rest_epochs : TYPE
        DESCRIPTION.

    Returns
    -------
    squared_event_epochs : TYPE
        DESCRIPTION.
    squared_rest_epochs : TYPE
        DESCRIPTION.

    '''
    
    # simply square the event epochs and the rest epochs
    squared_event_epochs=event_epochs**2
    squared_rest_epochs=rest_epochs**2
    
    # return the data
    return squared_event_epochs, squared_rest_epochs

# %% Baseline

# 4- Within each epoch, take a window near the end to use as a baseline
def get_baselines(data, squared_rest_epochs):
    '''
    use the end of each epoch sample as a baseline for information. Get the 
    baseline using the mean of each channel of each page. 

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    squared_rest_epochs : TYPE
        DESCRIPTION.

    Returns
    -------
    baselines : TYPE
        DESCRIPTION.

    '''
    
    # Get indexes representing last 1 sec worth of rest data in an epoch
    # where A is the start of that last second and B is the end
    B = np.shape(squared_rest_epochs)[1]
    A = B-1
    
    # get the number of pages and channels to loop through
    num_pages = np.shape(squared_rest_epochs)[0]
    num_channels = np.shape(squared_rest_epochs)[1]
    
    # create empty array to store the data. 
    # num_pages x num_channels, should be 34 x 32 if using default values
    baselines=np.zeros((num_pages,num_channels))
    
    # for each page, calculate the mean of the channels and save to array
    for r_index in range(len(squared_rest_epochs)): 
        baselines[r_index,:]=np.mean(squared_rest_epochs[r_index, A:B, :], axis=0)
        
    # return the data
    return baselines

# %% Subract Baseline from Squared Entries

def subract_baseline(squared_event_epochs, squared_rest_epochs, baselines):
    '''
    from the squared epochs and using the baseline information found earlier, 
    subtract the baseline data from all the epoch channels. 

    Parameters
    ----------
    squared_event_epochs : TYPE
        DESCRIPTION.
    squared_rest_epochs : TYPE
        DESCRIPTION.
    baselines : TYPE
        DESCRIPTION.

    Returns
    -------
    events_minus_baseline : TYPE
        DESCRIPTION.
    rests_minus_baseline : TYPE
        DESCRIPTION.

    '''
    
    # subtract baseline for the event epochs

    # create the array
    events_minus_baseline=np.zeros(np.shape(squared_event_epochs))
    # for each epoch, subtract out the baseline for the channels
    for index in range(len(squared_event_epochs)): 
        events_minus_baseline[index,:,:]=squared_event_epochs[index,:,:]-baselines[index,:]
    
    # subtract baseline for the rest epochs
    
    # create the array
    rests_minus_baseline=np.zeros(np.shape(squared_rest_epochs))
    # for each epoch, subtract out the baseline for the channels
    for index in range(len(squared_rest_epochs)): 
        rests_minus_baseline[index,:,:]=squared_rest_epochs[index,:,:]-baselines[index,:]
    
    # return the data
    return events_minus_baseline, rests_minus_baseline

# %% GET MEAN AND STANDARD ERROR

def get_mean_SE(events_minus_baseline, rests_minus_baseline):
    '''
    Calculate the mean and standard error for the event and the rest

    Parameters
    ----------
    events_minus_baseline : TYPE
        DESCRIPTION.
    rests_minus_baseline : TYPE
        DESCRIPTION.

    Returns
    -------
    mean_events : TYPE
        DESCRIPTION.
    mean_rests : TYPE
        DESCRIPTION.
    events_se : TYPE
        DESCRIPTION.
    rests_se : TYPE
        DESCRIPTION.

    '''
    # Get the mean signal across all epochs for each channel
    mean_events=np.mean(events_minus_baseline, axis=0)
    mean_rests=np.mean(rests_minus_baseline, axis=0)
    
    # Get the standard deviation of the signal across all epochs for each channel
    events_std = np.std(events_minus_baseline, axis=0)
    rests_std = np.std(rests_minus_baseline, axis=0)
    
    # Get the standard error of the the signal across all epochs for each channel
    events_se = events_std/np.sqrt(np.shape(events_minus_baseline)[1])
    rests_se = rests_std/np.sqrt(np.shape(rests_minus_baseline)[1])

    # return everything
    return mean_events, mean_rests, events_se, rests_se

# %% PLOT RESULTS 

def plot_results(mean_events, mean_rests, events_se, rests_se, data, epoch_duration, channels_to_plot, DEFAULT_SUBJECT, DEFAULT_SERIES):
    '''
    Plot the data for electrodes given. plot both mean +/- std error for the channels during the event and 
    at rest. Save both figures


    Parameters
    ----------
    mean_events : TYPE
        DESCRIPTION.
    mean_rests : TYPE
        DESCRIPTION.
    events_se : TYPE
        DESCRIPTION.
    rests_se : TYPE
        DESCRIPTION.
    data : TYPE
        DESCRIPTION.
    epoch_duration : TYPE
        DESCRIPTION.
    channels_to_plot : TYPE
        DESCRIPTION.
    DEFAULT_SUBJECT : TYPE
        DESCRIPTION.
    DEFAULT_SERIES : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    # Get epoch times (in seconds)
    epoch_times = np.arange(0,epoch_duration,1/data['fs'])
    
    # Extract channels to plot
    channel1 = np.where(data['channels'] == channels_to_plot[0])[0][0]
    channel2 = np.where(data['channels'] == channels_to_plot[1])[0][0]
    
    # Create figure
    plt.figure('events')
    plt.clf()
    
    # Plot mean_events over time for channels 1 and 2
    plt.plot(epoch_times, mean_events[:, channel1], label=f'{channels_to_plot[0]} Means')
    plt.plot(epoch_times, mean_events[:, channel2], label=f'{channels_to_plot[1]} Means')
    
    # Get upper limit for channel 1
    channel1_event_UL = mean_events[:, channel1] + events_se[:, channel1]
    # Get lower limit for channel 1
    channel1_event_LL = mean_events[:, channel1] - events_se[:, channel1]
    # Get upper limit for channel 2
    channel2_event_UL = mean_events[:, channel2] + events_se[:, channel2]
    # Get lower limit for channel 2
    channel2_event_LL = mean_events[:, channel2] - events_se[:, channel2]
    
    # plot CI for event means using mean +/- stderr for channels 1 and 2
    plt.fill_between(epoch_times, channel1_event_UL, channel1_event_LL, alpha=0.5, lw=5, label=f'{channels_to_plot[0]} +/- Standard Error')
    plt.fill_between(epoch_times, channel2_event_UL, channel2_event_LL, alpha=0.5, lw=5, label=f'{channels_to_plot[1]} +/- Standard Error')
    
    # Annotate plot
    plt.title(f'Events: Event Related (De)synchronization for Subject {DEFAULT_SUBJECT} Series {DEFAULT_SERIES}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Voltage (V)')
    plt.legend()
    
    # Make it look nice :)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('events.png')
    
    # Repeat the above for the rest epochs
    # Create figure
    plt.figure('rests')
    plt.clf()
    
    # Plot mean_events over time for channels 1 and 2
    plt.plot(epoch_times, mean_rests[:, channel1], label=f'{channels_to_plot[0]} Means')
    plt.plot(epoch_times, mean_rests[:, channel2], label=f'{channels_to_plot[1]} Means')
    
    # Get upper limit for channel 1
    channel1_rest_UL = mean_rests[:, channel1] + rests_se[:, channel1]
    # Get lower limit for channel 1
    channel1_rest_LL = mean_rests[:, channel1] - rests_se[:, channel1]
    # Get upper limit for channel 2
    channel2_rest_UL = mean_rests[:, channel2] + rests_se[:, channel2]
    # Get lower limit for channel 2
    channel2_rest_LL = mean_rests[:, channel2] - rests_se[:, channel2]
    
    # plot CI for event means using mean +/- stderr for channels 1 and 2
    plt.fill_between(epoch_times, channel1_rest_UL, channel1_rest_LL, alpha=0.5, lw=5, label=f'{channels_to_plot[0]} +/- Standard Error')
    plt.fill_between(epoch_times, channel2_rest_UL, channel2_rest_LL, alpha=0.5, lw=5, label=f'{channels_to_plot[1]} +/- Standard Error')
    
    # Annotate plot
    plt.title(f'Rests: Event Related (De)synchronization for Subject {DEFAULT_SUBJECT} Series {DEFAULT_SERIES}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Voltage (V)')
    plt.legend()
    
    # Make it look nice :)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('rests.png')
    
