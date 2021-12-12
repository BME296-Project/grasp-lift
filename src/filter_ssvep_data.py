#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 11:57:29 2021

Authors: Anna Jane Brown and George Spearing
@george and @annajane for graduate credit

This script contains function to make a filter, use it to filter data, 
get the envelope of that filtered data, and compare multiple graphs

"""

# Import packages
import numpy as np
from matplotlib import pyplot as plt
import os
from scipy import fft
from scipy.signal import firwin, freqz, filtfilt, hilbert

# %% Part 1: Load the Data

# See test script - recycled module from lab #3

# %% Part 2: Design a Filter

def make_bandpass_filter(low_cutoff, high_cutoff, filter_order, fs, filter_type='hann'):
    '''
    This function makes a filter based on the parameters passed in and returns
    the coefficients for the filter.

    Parameters
    ----------
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
    filter_coefficients: np 1d array
        the coefficients of the filter

    '''
    
    # Create an empty array for filter_coefficients
    filter_coefficients = np.zeros(filter_order)
    
    # Create the FIR filter using scipy.firwin
    numtaps = filter_order + 1
    filter_coefficients = firwin(numtaps, cutoff=[low_cutoff, high_cutoff], fs=fs, window=filter_type, pass_zero='bandpass')
    
    midpoint = int((low_cutoff + high_cutoff)/2)
    
    # Use scipy.freqz to get the frequency response
    freqs, freq_response = freqz(b=filter_coefficients, fs=fs)
    
    # Convert freq_response to dB
    freq_response = 10*np.log10(np.abs(freq_response)**2)
    
    # Define filter time axis
    t_filter = np.arange(0,len(filter_coefficients)/fs, 1/fs)
    
    # Plot the frequency response and the impulse response of your filter
    # Create first subplot
    plt.figure(filter_type)
    plt.subplot(2,1,1)
    plt.grid()
    
    # Plot frequency response
    plt.plot(freqs, freq_response, label=f'{midpoint} Hz')
    
    # plot the vertical line at the middle of the filter
    # get the label color
    if midpoint == 12: 
        c = 'blue'
    else:
        c = 'orange'
    plt.vlines(midpoint, min(freq_response), max(freq_response), linestyles='dashed', label=f'{midpoint} Hz', colors = c)
    
    # Annotate top subplot
    plt.title(f'{filter_type} FIR Filter Frequency Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.legend()
    
    # Create second subplot
    plt.subplot(2,1,2)
    plt.grid()
    
    # Plot impulse response
    plt.plot(t_filter, filter_coefficients, label=f'{midpoint} Hz')
    
    # Annotate bottom subplot
    plt.title(f'{filter_type} FIR Filter Impulse Response')
    plt.xlabel('Delay (s)')
    plt.ylabel('Gain')
    plt.legend()
    
    # Make it look nice
    plt.tight_layout()

    # Change directory to where you want the figure saved (for author's purposes only)
    # os.chdir('/Users/annajanebrown/Downloads/Fourth Year/Brain Computer Interfaces/Labs/Lab 4 figures')
    
    # Save the figure
    plt.savefig(f'{filter_type}_filter_{low_cutoff}-{high_cutoff}Hz_order{filter_order}.png')
    
    return filter_coefficients
    
# %% Part 3: Filter the EEG Signals

def filter_data(data, filter_coefficients):
    '''
    This function filters a signal given the data and filter coefficients.

    Parameters
    ----------
    data : dictionary
        raw data and realted channels and data information.
    filter_coefficients: np 1d array
        the coefficients of the filter. size 1 x number of samples
        
    Returns
    -------
    filtered_data : np 2d array
        the filtered data through the filter. size num_channels x num_samples

    '''
    # Extract the raw data from the data dictionary
    raw_data = data['eeg']
    
    # Use scipy.filter.filtfilt to apply the filter forward/backward in time 
    # to each channel in the raw data    
    filtered_data= filtfilt(b=filter_coefficients, a=1, x=raw_data)
    
    return filtered_data
    
# %% Part 4: Calculate the Envelope

def get_envelope(data, filtered_data, channel_to_plot=None, isolated_freq='X Hz'):
    '''
    This function gets the envelope of the data using the filtered data.

    Parameters
    ----------
    data : dictionary
        raw data and realted channels and data information.
    filtered_data : np 2d array
        the filtered data through the filter. size num_channels x num_samples
    fs : int
        Sampling frequency of the data.
    channel_to_plot : string, optional
        which EEG channel to show (only one channel allowed per function call). The default is None.

    Returns
    -------
    envelope : Array of float64
        An array of envelopes of the filtered data for all channels

    '''
    
    # Use data to redefine dictionary entries as local variables
    fs=data['fs']
    eeg = data['eeg']
    
    # Define time
    time = np.arange(0,np.shape(eeg[1])/fs,1/fs)
    
    # Extract the array of envelopes
    envelope = np.abs(hilbert(filtered_data))
    # Convert volts to microvolts
    envelope = envelope * (10**6)
    
    # Create an if statement st if channel_to_plot is not None, the function will
    # create a new figure and plot the band-pass filtered data on the given
    # electrode with its envelope on top.  
    if channel_to_plot != None :
        channels = data['channels']
        electrode = (np.where(channels==channel_to_plot))[0][0]
    
    # Extract envelope for given electrode
    channel_envelope = envelope[electrode]
     
    # Use electrode to index fltered data
    filtered_channel_data = filtered_data[electrode]
    # Convert volts to microvolts
    filtered_channel_data = filtered_channel_data * (10**6)
    
    
    # Plot the band-pass filtered data for the given electrode with its envelope on top
    plt.figure(isolated_freq)
    plt.plot(time, filtered_channel_data, label='Filtered Signal')
    plt.plot(time, channel_envelope, label='Envelope')
    
    # Annotate plot
    plt.title(f'{isolated_freq} BPF Data')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (uV)')
    plt.legend()
    
    # Make it look nice :)
    plt.tight_layout()
    
    # Change directory to where you want the figure saved (for author's purposes only)
    # os.chdir('/Users/annajanebrown/Downloads/Fourth Year/Brain Computer Interfaces/Labs/Lab 4 figures')
    
    # Save the figure
    plt.savefig(f'{isolated_freq} BPF Data.png')
    
    # Return amplitude of oscillations (on every channel at every time point) in an array called envelope.
    return channel_envelope, envelope
    
    
# %% Part 5: Plot the Amplitudes

def plot_ssvep_amplitudes(subject, data, envelope_1, envelope_2, channel_to_plot, isolated_freq1, isolated_freq2):
    '''
    This function plots two things in two subplots (in a single column): 
    # 1) the event start & end times and
    # 2) the envelopes of the two filtered signals

    Parameters
    ----------
    data : dictionary
        raw data and realted channels and data information.
    envelope_1 : Array of float64
        The envelope of the filtered data we want.
    envelope_2 : Array of float64
        The envelope of the filtered data we want.
    channel_to_plot : string
        the channel name of what to plot. based on EEG electrode names
    isolated_freq1 : int
        The frequency of the first envelope (for labeling only)
    isolated_freq2 : int
        the frequency of the second envelope (labeling only)
    subject : int
        the number of the subject the data is for
        

    Returns
    -------
    None.

    '''
    
    # Use data to redefine dictionary entries as local variables
    eeg = data['eeg']
    channel = data['channels']
    fs = data['fs']
    event_samples = data['event_samples']
    event_durations = data['event_durations']
    event_types = data['event_types']
    
    # Define time
    time = np.arange(0,(np.shape(eeg)[1]/fs), 1/fs)
    
    # Create an if statement st if channel_to_plot is not None, the function will
    # create a new figure and plot the band-pass filtered data on the given
    # electrode with its envelope on top.  
    if channel_to_plot != None :
        channels = data['channels']
        electrode = (np.where(channels==channel_to_plot))[0][0]
    
    # Extract envelope for given electrode
    channel_envelope_1 = envelope_1[electrode]
    channel_envelope_2 = envelope_2[electrode]
          
    # Create a figure with 2 subplots
    figure, axes = plt.subplots(2, sharex=True)
    
    # Plot the events as horizontal lines with dots at the start and end times
    for event_index, event_value in enumerate(event_samples):
        start_time = time[event_value]
        end_time = start_time + (event_durations[event_index]/fs)
        axes[0].plot([start_time, end_time], [event_types[event_index], event_types[event_index]], '-bo', markersize = 2) #event_duration, event_types
    plt.grid()
    
    # Annotate top subplot
    axes[0].set_ylabel(f'Subject {subject} SSVEP Amplitudes')
    axes[0].set_title(f'SSVEP Subject {subject} Raw Data')
    
    # Plot voltage of Fz and Oz electrodes over time
    axes[1].plot(time, channel_envelope_1, label=f'{isolated_freq1}')
    axes[1].plot(time, channel_envelope_2, label=f'{isolated_freq2}')
    
    # Annotate bottom subplot
    axes[1].set_title('Envelope Comparison')
    axes[1].set_ylabel('Voltage (uV)')
    axes[1].legend()
    
    #Annotate the plot
    plt.xlabel('Time (s)')
    
    # Make it look nice
    figure.tight_layout()
    
    # Save the figure
    #os.chdir('/Users/annajanebrown/Downloads/Fourth Year/Brain Computer Interfaces/Labs/Lab 4 figures') #for author only
    plt.savefig(f'SSVEP_S{subject}_{channel_to_plot}_{isolated_freq1}_and_{isolated_freq2}_Envelopes.png')

    return

