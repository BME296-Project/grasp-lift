#!/usr/bin/env python3
"""
main_grasp_lift.py

Main driver file for the grasp lift function library. 
Load in the data, epoch all the data, filter results, run ML
classification and make predictions of actions


@author: george
"""

# %% IMPORTS
import loadEEGData as eegdata
import helper_grasp_lift as gl

# %% CONSTANTS

DEFAULT_DIRECTORY = '../train' # default relative location to the training data
DEFAULT_SUBJECT = 1 # which subject to use, 1:12
DEFAULT_SERIES = 2 # which series to use, 1:8

DEFAULT_EPOCH_DURATION = 1 # how long the epoch is in seconds
DEFAULT_CHANNELS_CLASSIFICATION = ['Fp1', 'Fp2', 'Cz']


#%% LOAD IN DATA

data = eegdata.loadData(subject=DEFAULT_SUBJECT, series=DEFAULT_SERIES, data_directory=DEFAULT_DIRECTORY)

#%% FILTER THE DATA

filtered_data = gl.filter_data(data)

#%% EPOCH THE DATA

start_times, end_times, eeg_epochs, epoch_duration = gl.epoch_data(data)

#%% SQUARE AND BASELINE

squared_epochs=gl.square_epoch(eeg_epochs)
baseline=gl.baseline_epoch(squared_epochs)
subtracted_baseline, mean_baseline=gl.subract_baseline(squared_epochs, baseline)

#%% PLOT RESULTS

channels_to_plot=['C3', 'C4']
epoch_times=gl.plot_mean(mean_baseline, data, epoch_duration, channels_to_plot)

#%% RESULTS