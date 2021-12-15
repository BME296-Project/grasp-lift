#!/usr/bin/env python3
"""
main_grasp_lift.py

Main driver file for the grasp lift function library. 
Load in the data, filter data, create epochs, square and get baseline for data, 
get the mean and standard error of the data, plot results.


@author: @author: anna jane, brendan, george, jason
"""

# %% IMPORTS
import loadEEGData as eegdata
import helper_grasp_lift as gl

# %% CONSTANTS

DEFAULT_DIRECTORY = '../train' # default relative location to the training data
DEFAULT_SUBJECT = 1 # which subject to use, 1:12
DEFAULT_SERIES = 2 # which series to use, 1:8

DEFAULT_EPOCH_DURATION = 1 # how long the epoch is in seconds
DEFAULT_CHANNELS_CLASSIFICATION = ['C3', 'C4']


#%% LOAD IN DATA

data = eegdata.loadData(subject=DEFAULT_SUBJECT, series=DEFAULT_SERIES, data_directory=DEFAULT_DIRECTORY)

#%% FILTER THE DATA

filtered_data = gl.filter_data(data)

#%% EPOCH THE DATA

start_times, end_times, buffered_start_times, buffered_end_times, event_epochs, \
    rest_epochs, epoch_duration = gl.epoch_data(data, filtered_data)

#%% SQUARE AND BASELINE

squared_event_epochs, squared_rest_epochs = gl.square_epoch(event_epochs, rest_epochs)
baselines = gl.get_baselines(data, squared_rest_epochs)
events_minus_baseline, rests_minus_baseline = gl.subract_baseline(squared_event_epochs, squared_rest_epochs, baselines)

#%% GET MEAN AND STANDARD ERROR
mean_events, mean_rests, events_se, rests_se = gl.get_mean_SE(events_minus_baseline, rests_minus_baseline)

#%% PLOT RESULTS
# Specify channels to plot
channels_to_plot=DEFAULT_CHANNELS_CLASSIFICATION

gl.plot_results(mean_events, mean_rests, events_se, rests_se, data, epoch_duration, \
                channels_to_plot, DEFAULT_SUBJECT, DEFAULT_SERIES)
