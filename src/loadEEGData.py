#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 14:43:03 2021

@author: george

# TODO: Add section to import the truth data from the subject
"""

import numpy as np

def loadData(subject=1, series=1, data_directory='train'):
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
    
    file_path = f'{data_directory}/subj{subject}_series{series}_data.csv'
    
    # load just the data
    eegData = np.loadtxt(file_path, delimiter=',', usecols=(np.arange(1,33)), skiprows=1)
    
    # just the channel names
    channels = np.loadtxt(file_path, delimiter=',', usecols=(np.arange(1,33)), max_rows=1, dtype=str, unpack=True)
    
    # Load in truth Data corresponding to EEG Events

    # turn it into a dictionary
    data = {}
    data['eeg'] = eegData
    data['channels'] = channels

    # return the data
    return data