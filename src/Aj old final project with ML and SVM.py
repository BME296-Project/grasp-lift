# -*- coding: utf-8 -*-
"""
Created on Thu May  6 18:43:52 2021

BradYoung_Project_3_B.py

-This code will load in data that was given to us by Dr. Jangaw as well as data created for the purpose
of this project. The data will be put into epochs and the features will be extracted for the variance
of each channel (total 3). There are 4 different ways accuracy was tested to be improved. These include: 
Using a linear kernan in GridSearchCV, setting the decision function shape in GridSearchCV to be ovo,
Training the classifier on data created just for this project (not given to us), and lastly by attempting
to use KNeighborsClassifier at 6 different number of neighbor values to see where the accuracy was the
highest. Finally, the best classifier observed is saved to be used in part C of this project.

@author: Brad Young

Edited 5/12/21 finalized code, added comments, determined one last way to try and improve accuracy.


"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import LoadAndPlotRpsData as lap
import BradYoung_Lab5_modified as lab5
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


# %% Load datasets
participant_list = ['A', 'B', 'C', 'D', 'E', 'F','G','H']
action_sequence = np.array(['Rest', 'Rock', 'Rest', 'Paper', 'Rest', 'Scissors']*10)
emg_data_list = lap.load_rps_data(in_folder = '.', participant_list = participant_list, out_folder = '.', action_sequence = action_sequence)

#Load my recorded data
b_data = np.load('C:/Users/byoun/Documents/Biomedical Instrumentation/Labs/Projects/Project 3/Young_2_RPS_Data.npy')
# %% Extract and Scale

#Classes we want to keep
classes_to_keep = ['Paper', 'Scissors', 'Rock', 'Rest']

#Initialize feature and truth lists
features_ps_list = []
truth_ps_list = []

#Epoch loaded data (my recorded data)
b_emg_epoch = lab5.epoch_data(b_data, fs =500, epoch_duration = 1)

#Extract features and feature names from b_emg_epoch
b_features, feature_names = lab5.extract_features(b_emg_epoch) 



for participant_index, emg_data in enumerate (emg_data_list):
    
    #extract epoch
    emg_epoch = lab5.epoch_data(emg_data, fs =500, epoch_duration = 1)
    
    #extract features 
    features, feature_names = lab5.extract_features(emg_epoch)
    
   #Attempt to use squared emg data instead of variance 
    #features, feature_names = lab5.extract_features_2(emg_epoch)
      
    #crop to classes we want
    features_ps, truth_labels_ps = lab5.crop_ml_inputs(features, action_sequence, classes_to_keep)
        
    #add features and truth labels to list
    features_ps_list = features_ps_list + [features_ps]
    truth_ps_list = truth_ps_list + [truth_labels_ps]
           
# Split data into train and test sets
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(b_features, truth_labels_ps, test_size = 0.25)


#combine accross participants
features_ps = np.vstack(features_ps_list)
truth_labels_ps = np.hstack(truth_ps_list)

X_train, X_test, y_train, y_test = train_test_split(features_ps, truth_labels_ps, test_size = 0.25)

#Initialize an array of C values to be used in a Grid search
C_array = np.array([1e-2,1e0,1e2, 1e4, 1e6, 1e8, 1e10])
#Initialize gamma
gamma = 10

#Declare our hyperparameter grid
param_grid = {'C':C_array}

# %% Cross validated grid search control 

#Create cross-validated grid search object
clf_control= GridSearchCV(SVC(gamma = gamma), param_grid, scoring='accuracy')
#Perform cross-validated grid search on our training set
print('Tuning hyperparameters...')
clf_control.fit(X_train,y_train)
print('Done!')

# %% Plot performance accuracy of Classifier #1

plt.figure(4)
# extract info from fit grid search 
mean_accuracy = clf_control.cv_results_['mean_test_score']
# print(mean_accuracy)
plt.plot(C_array,mean_accuracy,'.-')
plt.xscale('log')
plt.xlabel('C')
plt.ylabel('Mean CV Accuracy')



# %% Save trained classifier #1 and print accuracy

clf_best_c = clf_control.best_estimator_
clf_best_c.predict(X_test_b)
y_prime = clf_control.predict(X_test_b)
accuracy_control = clf_best_c.score(X_test_b,y_test_b)
print(f'accuracy = {accuracy_control}')

# %% Cross validated grid search #1 (Change kernel to linear)

#Create cross-validated grid search object
clf_linear = GridSearchCV(SVC(gamma = gamma, kernel = 'linear' ), param_grid, scoring='accuracy')
#Perform cross-validated grid search on our training set
print('Tuning hyperparameters...')
clf_linear.fit(X_train,y_train)
print('Done!')

# %% Plot performance accuracy of Classifier #1

plt.figure(0)
# extract info from fit grid search 
mean_accuracy_linear = clf_linear.cv_results_['mean_test_score']
# print(mean_accuracy)
plt.plot(C_array,mean_accuracy_linear,'.-')
plt.xscale('log')
plt.xlabel('C')
plt.ylabel('Mean CV Accuracy')

# %% Save trained classifier #1 and print accuracy

clf_best_linear = clf_linear.best_estimator_
clf_best_linear.predict(X_test_b)
y_prime = clf_linear.predict(X_test_b)
accuracy_linear = clf_best_linear.score(X_test_b,y_test_b)
print(f'accuracy linear = {accuracy_linear}')

# %% Cross validated grid search #2 (Change decison function shape to one vs one)

# Create cross-validated grid search object
clf_ovo = GridSearchCV(SVC(gamma = gamma, decision_function_shape = 'ovo'), param_grid, scoring='accuracy', cv = 6)
# Perform cross-validated grid search on our training set
print('Tuning hyperparameters...')
clf_ovo.fit(X_train,y_train)
print('Done!')

# %% Plot performance accuracy of Classifier #2

#Initialize figure
plt.figure(1)
# extract info from fit grid search 
mean_accuracy_ovo = clf_ovo.cv_results_['mean_test_score']
# print(mean_accuracy)
plt.plot(C_array,mean_accuracy_ovo,'.-')
plt.xscale('log')
plt.xlabel('C')
plt.ylabel('Mean CV Accuracy')

# %% Save trained classifier #2 and print accuracy

clf_best_ovo = clf_ovo.best_estimator_
clf_best_ovo.predict(X_test_b)
y_prime = clf_ovo.predict(X_test_b)
accuracy_ovo = clf_best_ovo.score(X_test_b,y_test_b)
print(f'accuracy one vs one = {accuracy_ovo}')

# %% Cross validated grid search #3 (Trained on my recorded data)

# Create cross-validated grid search object
clf_b = GridSearchCV(SVC(gamma = gamma), param_grid, scoring='accuracy', cv = 6)
# Perform cross-validated grid search on our training set
print('Tuning hyperparameters...')
clf_b.fit(X_train_b,y_train_b)
print('Done!')

# %% Plot performance accuracy of Classifier #3

#Initialize figure
plt.figure(2)
# extract info from fit grid search 
mean_accuracy_b = clf_b.cv_results_['mean_test_score']
# print(mean_accuracy)
plt.plot(C_array,mean_accuracy_b,'.-')
plt.xscale('log')
plt.xlabel('C')
plt.ylabel('Mean CV Accuracy')

# %% Save trained classifier #3 and print accuracy

clf_best_b = clf_b.best_estimator_
clf_best_b.predict(X_test)
y_prime = clf_b.predict(X_test)
accuracy_b = clf_best_b.score(X_test,y_test)
print(f'accuracy when trained on other data = {accuracy_b}')

# %% Classifier modification #4 (Try K Neighbors)
def try_kn():
    #Initializes accuracy and classifier names variables
    mean_accuracy_kn = np.zeros(5)
    kn_names = ['']*5
    
    #for loop that tries kNeighbors at 6 different neightbors values
    for n_neighbors in range(5):
        
        classifier_name= 'clf_kn_'+ f'{n_neighbors}'
        # Create cross-validated grid search object
        classifier_name = KNeighborsClassifier(n_neighbors = n_neighbors+1, weights = 'uniform')
        # Perform cross-validated grid search on our training set
        print('Tuning hyperparameters...')
        classifier_name.fit(X_train,y_train)
        print('Done!')
        
        mean_accuracy_kn[n_neighbors] = classifier_name.score(X_test,y_test)
        print(f'accuracy of K Neighbors classifier looking for {n_neighbors} neighbors = {mean_accuracy_kn[n_neighbors]}')
        kn_names[n_neighbors] = classifier_name
    return mean_accuracy_kn, kn_names
mean_accuracy_kn, kn_names = try_kn()   
# %% Plot performance accuracy of Classifier #4
x_vals = [1,2,3,4,5]
#Initialize figure
plt.figure(3)

# print(mean_accuracy)
plt.plot(x_vals,mean_accuracy_kn,'.-')
plt.xlabel('number of neighbors (N)')
plt.ylabel('Mean Accuracy')
plt.title('Accuracy vs N (number of neighbors)')


# %% Compare all the accuracy

#List of all accuracy values
all_accuracy = [accuracy_control, accuracy_b, accuracy_ovo, accuracy_linear, mean_accuracy_kn[0]
                , mean_accuracy_kn[1], mean_accuracy_kn[2], mean_accuracy_kn[3], mean_accuracy_kn[4]]

#List of all classifiers
all_classifiers = [clf_best_c, clf_best_b, clf_best_ovo, clf_best_linear, kn_names[0]
                , kn_names[1], kn_names[2], kn_names[3], kn_names[4]] 

#Finds best accuracy
max_val = max(all_accuracy)
max_index = all_accuracy.index(max_val)
print(f'The best classifier is {all_classifiers[max_index]} and it has an accuracy of {all_accuracy[max_index]}')

#Makes the best accuracy the best classifier to use
clf_best = all_classifiers[max_index]

# %% Plot confusion matrices

#Plots confusion matrix using clf as the classifier and the training data as the input
# and target values 
confusion_matrix = metrics.plot_confusion_matrix(clf_best,X_test, y_test)
#Labels colorbar
confusion_matrix.im_.colorbar.set_label('Number of trials')
plt.title('Best Classifier on Test Data')
plt.show()
plt.savefig('Project_3_confusion_matrix.png')



# %% Part 6: Save the trained classifier object
from joblib import dump

 

# Save the trained classifier for use later (PART C of Project)
dump(clf_best,'BestClassifierEver.joblib')




                                                    