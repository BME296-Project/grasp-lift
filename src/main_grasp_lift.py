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
DEFAULT_SERIES = 1 # which series to use, 1:8

#%% LOAD IN DATA

data = eegdata.loadData(subject=DEFAULT_SUBJECT, series=DEFAULT_SERIES, data_directory=DEFAULT_DIRECTORY)

#%% EPOCH THE DATA

eeg_epoch, truth_epoch = gl.epoch_data(data['eeg'], data['truth_data'])

#%% FILTER THE DATA

#%% CLASSIFICATIONS

#%% RESULTS