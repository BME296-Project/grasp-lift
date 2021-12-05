# README for Grasp and Lift EEG Detection Challenge as a class project for BME296 at UVM

Written by group: Anna Jane, Brendan, George, Jason | 12/01/2021

## Data Information

This dataset can be used to analyze the EEG data from participants during arm motion events. The original dataset can be found at [https://www.kaggle.com/c/grasp-and-lift-eeg-detection/code](https://www.kaggle.com/c/grasp-and-lift-eeg-detection/code), with the experiment paper located here: [https://www.nature.com/articles/sdata201447](https://www.nature.com/articles/sdata201447). 

The data comes from 32 EEG probes on the scalp of 12 participants. There are 6 unique events that each participant had to complete. These included hand start, first touch, start load, lift off, replace, release. Sampling rate was 500Hz.

The object was lifted and held for 2 seconds. After the replace and release events, there was a 1-3 second rest before the next sequence of events. The weight and contact surface for the object changed at specific intervals, unknown to the participants. This results in specific modifications in musclecoordination to properly complete the event trial.

The data from the participants have already been broken down into training and test data sets. The “train” folder contains all the data while the “test” folder is 2 series trials from each participant.
In the training set, there are two files for each subject + series combination:
 - the *_data.csv files contain the raw 32 channels EEG data (sampling rate 500Hz)

 - the *_events.csv files contain the ground truth frame-wise labels for all events

## Script Information to Load Data

To load the data files, you can use the [loadEEGData.py](src/loadEEGData.py) file. The function “loadData” will import the eeg data
and eeg channel names into a dictionary. The data can be loaded into a dictionary using the following code:

    Import loadEEGData as ld
    data = ld.loadData(subject=1, series=1, data_directory=”train”)

The data filed can be extracted by using dictionary keys. For example:

    eegData = data[‘eeg’] # returns the raw eeg data into a variable

The dictionary keys are as follows:

**“eeg”** - the raw eeg data from the participant. All 32 channels arranged in a trials x channels np array. Data is a float type

**“channels”** - the eeg channel names in a 1d list of strings.

**“truth_events”** - The truth labels representing the truth of the event during a given event. Boolean values for each of the 6 events for each trial. Size of array is trials x events.
