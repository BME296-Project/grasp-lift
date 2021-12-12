#!/usr/bin/env python3

# George Spearing
# Brad Young
# GeorgeSpearing_Lab5.py
# April 2021
#
# Lab5 - Mahcine Learning Features
# Cell 1 - Capture Data (uses project 2 code base)
# Cell 2 - Extract Features for ML training / prediction
# cell 3 - Show data comparisons between features
# cell 4 - Create Classifier from training data
# cell 5 - Evaluate Performance of predcitions of classifier
# cell 6 - Test classifier on new data set
#

import numpy as np
from matplotlib import pyplot as plt
import os
from sklearn import preprocessing
from sklearn import svm
from sklearn import metrics

 # code from project 2 base


# global Definitions -- used for base case
#DEFAULT_BAUD_RATE = 500000 # arduino baudrate (kps)
#DEFAULT_PORT = '/dev/ttyUSB0' # arduino port default for George Computer
#DEFAULT_UPPER_VOLTAGE = 5 # arduino is 5V max
#DEFAULT_SIGNAL_BITS = 1024 # bits for signal to voltage conversion
#DEFAULT_MS_TO_SEC = 1000.0 # convert ms to s
#MAX_FAIL_ATTEMPS = 3 # num failed connections (attemps)
#SAMPLES_TO_THROW = 5 # num samples to discard at start of program
DEFAULT_EPOCH_DURATION = 1 # seconds for each epoch

DEFAULT_NUM_CHANNELS = 3
DEFAULT_FS = 500
#DEFAULT_OUT_FOLDER = "Lab5Data" # create new file to store data
#DEFAULT_OUT_FILENAME = "Spearing"
DEFAULT_DURATION = 60
verbose = False # for debugging -- will show print statements (slower)

# define actions taht are expected of the user in the order of operation
possible_actions = ['rest', 'rock','rest','paper','rest','scissors']


'''
Extract Features and Labels
'''
# create epoching of data set
def epoch_data(emg_data, fs, epoch_duration):
    '''
    Create epoch of data set
    @param emg_data, data values from cell1
    @param fs, sampling frequency
    @param epoch_duration, length of epoch we're doing
    '''
    # subtract out the mean
    mean_values = np.mean(emg_data,axis=0)
    dataSet = emg_data - mean_values

    # define values based on defaults
    epochs_number = ((DEFAULT_DURATION)*(epoch_duration))
    epoch_sample_count = fs
    channel_count = DEFAULT_NUM_CHANNELS
    emg_data_epoch = np.reshape(dataSet,(epochs_number, epoch_sample_count,channel_count))

    # return array of size: 
    # [epoch_count, epoch_sample_count, channel_count]
    return emg_data_epoch
    

# extract features from data set
def extract_features(epoch_data):
    '''
    extract 3 different features for each channel
    '''
    ###### define Features for later use
    features = ['var_ch0', 'var_ch1','var_ch2']

    # initialize new array to hold features
    feature_data = np.zeros([epoch_data.shape[0],len(features)])

    # extract features variance
    feature_data[:,0:3] = np.var(epoch_data, axis=1)
    
    # sklearn preprocessing
    preprocessing.scale(feature_data)
    preprocessing.normalize(feature_data)
    

    # return values for data and column labels
    return feature_data, features


# crop the features and labls to include only exmples in list
def crop_ml_inputs(trial_features, truth_labels_full, truth_search):
    '''
    Parse data for a subset of feature values and truth labels
    '''
    # get index values
    parse_data = np.where(np.in1d(truth_labels_full, truth_search))

    # parse features and labels based on search values
    features_ps = trial_features[parse_data]
    truth_labels_ps = truth_labels_full[parse_data]

    # return feature values, truth labels
    return features_ps, truth_labels_ps


    ######################
    # RUN ALL THE THINGS #
    #       CELL 2       #
    ######################

    # get data from cell1 file output
    #emg_data, emg_time = load_data(DATA_TO_TRAIN)
    #emg_epoch = epoch_data(emg_data, DEFAULT_FS, epoch_duration=1)
    #feature_data, features = extract_features(emg_epoch)
    #truth_data = make_truth_data(possible_actions, DEFAULT_EPOCH_DURATION)
    #truth_search = ['paper','scissors']
    #features_parsed, truth_label_parsed = crop_ml_inputs(feature_data, truth_data, truth_search)



# ###############
# #    CELL3    #
# ###############

# # def cell3(): # cannot be it's own function b/c we need values from cell 2

#     def scatter_plot(feature_matrix, feature_labels, truth_matrix, fileNameToSave):
#         '''
#         Plot features of various analysis, Variance, 
#         Mean Absolute Value, and Zero Crossings 
#         '''
        
#         # plot feature matrix values
#         plt.scatter(feature_matrix[:,0],feature_matrix[:,1])
#         plt.title(f'{feature_labels[0]} vs {feature_labels[1]} for {fileNameToSave}')
#         plt.xlabel(feature_labels[0])
#         plt.ylabel(feature_labels[1])
        
#         # save plot
#         plt.savefig(f'{DEFAULT_OUT_FOLDER}/ScatterPlot_{fileNameToSave}_{feature_labels[0]}.png')

#         # show plot to user
#         plt.show()


#     ######################
#     # RUN ALL THE THINGS #
#     #       CELL 3       #
#     ######################

#     # separate out data for function calls
#     var_ch1_ch2 = feature_data[:,(1,2)]
#     mav_zc_ch0 = feature_data[:,(3,6)]
#     var_ch0_zc_ch2 = feature_data[:,(0,7)]

#     # create three scatter plots
#     scatter_plot(var_ch1_ch2, [features[1],features[2]], truth_data, DATA_TO_TRAIN)
#     scatter_plot(mav_zc_ch0, [features[3],features[6]], truth_data, DATA_TO_TRAIN)
#     scatter_plot(var_ch0_zc_ch2, [features[0], features[7]], truth_data, DATA_TO_TRAIN)
#     #######################
#     # Part 3 Plot Comment 
#     # The variance between channel 1 and channel 2 has the strongest correlation.
#     # When the muscles move in the arms, there will be overlaping signals which
#     # would cause nearby muscles to also show a value. Some actions require multiple
#     # muscle movements which show similar variance values on different channels
#     #######################
    


# ###############
# #    CELL4    #
# ###############

# # def cell4():# cannot be it's own function b/c we need values from cell 2
    
#     def fit_classifier(features, labels):
#         '''
#         Create classifier from a subset of data 
#         @param features, subset of data to train classifier
#         @param labels, truth values to compare against
#         '''

#         # Bias labels to int values
#         scissors_true = labels=='scissors'
#         paper_true = labels=='paper'
#         bias_label=np.zeros(len(labels))
#         bias_label[scissors_true] = 1 # change scissors to 1
#         bias_label[paper_true] = -1 # change paper to -1
#         # convert to int values
#         bias_label = np.array(bias_label).astype(int)

#         # choose a linear classifier
#         clf = svm.LinearSVC(C=1e6, dual=False)

#         # Train the classifier to our data
#         clf.fit(features,bias_label)

#         # Use the classifier to get predictions for our data
#         y_prime = clf.predict(features)

#         # look at results of classifier prediction
#         w = clf.coef_ # weight vector
#         b = clf.intercept_ # bias term
#         z_prime = features @ w.T + b # predictor z' = X*wT + b
#         print(f'weights: {w}')
#         print(f'bias: {b}')
#         print(f'predictor: {z_prime}')
#         print(f'predicted labels: {y_prime}')
#         print(f'true labels: {bias_label}')

#         # return classifier object
#         return clf

#     ######################
#     # RUN ALL THE THINGS #
#     #       CELL 4       #
#     ######################

#     # run the classifier 
#     clf = fit_classifier(features_parsed, truth_label_parsed)


# ###############
# #    CELL5    #
# ###############

# # def cell5(): # cannot be it's own function b/c we need values from cell 2

#     def predictor_histogram(classifier, features, truth_labels, fileNameToSave):
#         '''
#         Plot the histogram of data to show paper vs rock variance
#         '''

#         # calculate predictor 
#         y_prime = classifier.predict(features)

#         # figure out the threshold value using the mean of variance of channel 0
#         y_line = np.abs((np.mean(features[y_prime==1,0]))-(np.mean(features[y_prime==-1,0])))

#         # plot histogram of data
#         plt.hist([features[y_prime==1,0],features[y_prime==-1,0]], alpha=0.5, bins=15, label=['paper','scissors'])
#         # plot vertical line
#         plt.axvline(x=y_line, color="red")

#         # make look pretty
#         plt.title(f'Classifications {fileNameToSave}')
#         plt.xlabel('Predictor')
#         plt.ylabel('Confidence')
#         plt.legend()

#         # save and show figure
#         plt.savefig(f'{DEFAULT_OUT_FOLDER}/Histogram_{fileNameToSave}.png')
#         plt.show()


#     def evaluate_classifier(classifier, features, truth_labels, fileNameToSave):
#         '''
#         Make a confusion matrix to show paper vs scissors prediction
#         '''

#         # generate predictions
#         y_prime = classifier.predict(features)

#         # Bias labels to int values
#         scissors_true = truth_labels=='scissors'
#         paper_true = truth_labels=='paper'
#         bias_label=np.zeros(len(truth_labels))
#         bias_label[scissors_true] = 1 # change scissors to 1
#         bias_label[paper_true] = -1 # change paper to -1
#         # convert to int values
#         bias_label = np.array(bias_label).astype(int)
        
#         # make stuff happen
#         tp_paper = (truth_labels=='paper') & (y_prime==-1)
#         fp_paper = ~(truth_labels=='paper') & (y_prime==-1)
#         fn_paper = (truth_labels=='paper') & ~(y_prime==-1)
#         tn_paper = ~(truth_labels=='paper') & ~(y_prime==-1)

#         tp_scissors = (truth_labels=='scissors') & (y_prime==1)
#         fp_scissors = ~(truth_labels=='scissors') & (y_prime==1)
#         fn_scissors = (truth_labels=='scissors') & ~(y_prime==1)
#         tn_scissors = ~(truth_labels=='scissors') & ~(y_prime==1)

#         # Accuracy
#         # accuracy (true positive + true negative / total samples)
#         accuracy_paper = (np.count_nonzero(tp_paper)+np.count_nonzero(tn_paper)) / (len(truth_labels))
#         accuracy_scissor = (np.count_nonzero(tp_scissors)+np.count_nonzero(tn_scissors)) / (len(truth_labels))

#         print(f'Accuracy Paper: {accuracy_paper}\n')
#         print(f'Accuracy Scissors: {accuracy_scissor}\n')

#         # plot confusion matrix
#         metrics.plot_confusion_matrix(classifier, features, bias_label, display_labels=['paper','scissors'])
#         plt.title(f'Confusion Matrix {fileNameToSave}')
#         plt.savefig(f'{DEFAULT_OUT_FOLDER}/Confusion_Matrix_{fileNameToSave}.png')
#         plt.show()


#     ######################
#     # RUN ALL THE THINGS #
#     #       CELL 5       #
#     ######################

#     predictor_histogram(clf, feature_data, truth_data, DATA_TO_TRAIN)
#     evaluate_classifier(clf, features_parsed, truth_label_parsed, DATA_TO_TRAIN)


# ###############
# #    CELL6    #
# ###############

# # def cell6():
#     '''
#     Run the classifier on a new data set
#     '''

#     # get data from cell1 file output
#     emg_data, emg_time = load_data(DATA_TO_PREDICT)
#     emg_epoch = epoch_data(emg_data, DEFAULT_FS, epoch_duration=1)
#     feature_data, features = extract_features(emg_epoch)
#     truth_data = make_truth_data(possible_actions, DEFAULT_EPOCH_DURATION)
#     truth_search = ['paper','scissors']
#     features_parsed, truth_label_parsed = crop_ml_inputs(feature_data, truth_data, truth_search)

#     # Scatter Plots
#     # separate out data for function calls
#     var_ch1_ch2 = feature_data[:,(1,2)]
#     mav_zc_ch0 = feature_data[:,(3,6)]
#     var_ch0_zc_ch2 = feature_data[:,(0,7)]

#     # create three scatter plots
#     scatter_plot(var_ch1_ch2, [features[1],features[2]], truth_data, DATA_TO_PREDICT)
#     scatter_plot(mav_zc_ch0, [features[3],features[6]], truth_data, DATA_TO_PREDICT)
#     scatter_plot(var_ch0_zc_ch2, [features[0], features[7]], truth_data, DATA_TO_PREDICT)

#     # predictor histogram and confusion matrix
#     predictor_histogram(clf, feature_data, truth_data, DATA_TO_PREDICT)
#     evaluate_classifier(clf, features_parsed, truth_label_parsed, DATA_TO_PREDICT)



# # run the code
# if __name__=="__main__":

#     # NOTE: UNCOMMENT WHICH CELL TO RUN, SEE NOTES

#     # cell1() # Record Data
#     cell2() # Extract Features and Labels
#     # cell3() # (plotting cell2 data) lIVES IN CELL2
#     # cell4() #  (Train Classifier) lIVES IN CELL2
#     # cell5() # Evaluate Training Performance lIVES IN CELL2
#     # cell6() # Test on New Data lIVES IN CELL2