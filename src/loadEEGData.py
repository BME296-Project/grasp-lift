#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 14:43:03 2021

@author: george

# TODO: Add section to import the truth data from the subject
"""
#%% 
import numpy as np

def loadData(subject=1, series=1, data_directory='../train'):
    '''
    load in one series from one subject. 

    Parameters
    ----------

    subject : int, optional
        The subject to analyze, value of 1:12 accepted, default is 1.
    series : int, optional
        The series to analyze, value of 1:8 accepted, default is 1.
    data_directory : string, optional
        The directory location for data, default is '../train'

    Returns
    -------
    data : Dictionary
        Dict of raw eeg values and channel names

    '''
    
    data_file_path = f'{data_directory}/subj{subject}_series{series}_data.csv'
    truth_file_path = f'{data_directory}/subj{subject}_series{series}_events.csv'
    
    # load just the data
    eegData = np.loadtxt(data_file_path, delimiter=',', usecols=(np.arange(1,33)), skiprows=1)
    
    # just the channel names
    channels = np.loadtxt(data_file_path, delimiter=',', usecols=(np.arange(1,33)), max_rows=1, dtype=str, unpack=True)
    
    # Load in truth Data corresponding to EEG Events
    truth_data = np.loadtxt(truth_file_path,delimiter=',', usecols=(np.arange(1,7)), skiprows=1, dtype=int)
    events = np.loadtxt(truth_file_path,delimiter=',', usecols=(np.arange(1,7)), max_rows=1, dtype=str)
    

    # turn it into a dictionary
    data = {}
    data['eeg'] = eegData
    data['channels'] = channels
    data['truth_data'] = truth_data
    data['events'] = events

    # return the data
    return data

#%% 
data = loadData()
# %%
