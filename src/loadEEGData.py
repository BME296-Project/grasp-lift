#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 14:43:03 2021

loadEEGData.py

single function to load the data from the subject and series information. 
Loads both the raw eeg data and the truth events. 
Returns all the information in a dictionary

@author: anna jane, brendan, george, jason


"""
# %% IMPORTS
import numpy as np


# %% LOADING DATA

def loadData(subject=1, series=1, data_directory='../train'):
    '''
    load in one series from one subject. loads both raw eeg and truth. 
    Input data is in the form of csv files

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
        Dict of raw eeg, chanels, truth data, events, and fs

    '''
    
    # create the file paths from the input parameters
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
    
    # known values from the experiment description
    data['fs'] = 500

    # return the data
    return data

# %% IMPORT CONTROL

if __name__ == '__main__':
    print("Import Only")