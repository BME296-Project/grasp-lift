#!/usr/bin/env python3

"""
helper_grasp_lift.py

functions to help with data analysis. this includes epoching
data, making classifications, predictors and plotting results

@author: george
"""

# %% IMPORTS


# %% EPOCH THE DATA

# Input: Raw EEG Data, start time, end time
# Output: 3d array of epoched events. Raw data turned into chunks of data
# NOTE: Look at experiement to see how it was set up. There may have been rough timing for each events. Then we can epoch around the center of that time window? 
# NOTE: Check data for start and end point of potnetial epochs -> compare across subject

#%% FILTERING

# Filter Data using fft things -> hilbert tranforms? scipy.filter.filtfilt()?
# Input: Raw EEG data
# Output: Filtered EEG Data

#%% CLASSIFICATIONS

# Generate or predict threshold values for the data. 
# Input: Epoched EEG Data, filtered, start event times
# Output: Predicted Events

# Taking in the filtered data and some channels (realistically we should focus on just one for now) that we want to look at
# Then generate a prediction based on a threshold of some sort

# Some variable names for predicted events
is_predicted_hand_start
is_predicted_first_touch
is_predicted_grasp
is_predicted_lift
is_predicted_replace
is_predicted_release

#%% RESULTS

# Input: Predicted Events, Truth Events
# Output: Confusion Matrix

# Confusion Matrix to compare the the predicted and truth events
# We can test using the given testing data things
# Accuracy metric and ITR (becuase ITR is simple and pretty)