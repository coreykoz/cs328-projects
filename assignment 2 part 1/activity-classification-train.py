# -*- coding: utf-8 -*-
"""
This is the script used to train an activity recognition 
classifier on accelerometer data.

"""

import os
import sys
import numpy as np
from sklearn import model_selection
#from sklearn.svm._libsvm import predict
from sklearn.tree import export_graphviz, DecisionTreeClassifier, DecisionTreeRegressor
from features import extract_features
from util import slidingWindow, reorient, reset_vars
import pickle
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, confusion_matrix, recall_score, accuracy_score

# %%---------------------------------------------------------------------------
#
#		                 Load Data From Disk
#
# -----------------------------------------------------------------------------

print("Loading data...")
sys.stdout.flush()
data_file = 'my-activity-data.csv'
data = np.genfromtxt(data_file, delimiter=',')
print("Loaded {} raw labelled activity data samples.".format(len(data)))
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                    Pre-processing
#
# -----------------------------------------------------------------------------

print("Reorienting accelerometer data...")
sys.stdout.flush()
reset_vars()
reoriented = np.asarray([reorient(data[i,1], data[i,2], data[i,3]) for i in range(len(data))])
reoriented_data_with_timestamps = np.append(data[:,0:1],reoriented,axis=1)
data = np.append(reoriented_data_with_timestamps, data[:,-1:], axis=1)

# %%---------------------------------------------------------------------------
#
#		                Extract Features & Labels
#
# -----------------------------------------------------------------------------

window_size = 20
step_size = 20

# sampling rate should be about 25 Hz; you can take a brief window to confirm this
n_samples = 1000
time_elapsed_seconds = (data[n_samples,0] - data[0,0]) / 1000
sampling_rate = n_samples / time_elapsed_seconds

# TODO: list the class labels that you collected data for in the order of label_index (defined in collect-labelled-data.py)
class_names = ["walking", "sitting", "standing", "stairs"]

print("Extracting features and labels for window size {} and step size {}...".format(window_size, step_size))
sys.stdout.flush()

X = []
Y = []

for i,window_with_timestamp_and_label in slidingWindow(data, window_size, step_size):
    window = window_with_timestamp_and_label[:,1:-1]   
    feature_names, x = extract_features(window)
    X.append(x)
    Y.append(window_with_timestamp_and_label[10, -1])
    
X = np.asarray(X)
Y = np.asarray(Y)
n_features = len(X)
    
print("Finished feature extraction over {} windows".format(len(X)))
print("Unique labels found: {}".format(set(Y)))
print("\n")
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                Train & Evaluate Classifier
#
# -----------------------------------------------------------------------------


# TODO: split data into train and test datasets using 10-fold cross validation
cv = model_selection.KFold(n_splits=10, random_state=None, shuffle=True)
tree = DecisionTreeClassifier(criterion="entropy", max_depth=3)

Y_tests = []
Y_preds = []
conf_matrixs = []
accuracy_vals = []
precision_vals = []
recall_vals = []
fold_count = 1

for train_index, test_index in cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    model = tree.fit(X_train, Y_train.astype(int))
    Y_Pred = model.predict(X_test)
    conf = confusion_matrix(Y_test, Y_Pred.astype(int))
    conf_matrixs.append(conf)
    Y_tests.append(Y_test)
    Y_preds.append(Y_Pred)

    tp = (conf[0][0] + conf[1][1] + conf[2][2] + conf[3][3])

    sum = 0
    for element in conf:
        for elem in element:
            sum += elem

    accuracy = (tp/sum) * 100
    precision = precision_score(Y_test, Y_Pred.astype(int), average="micro")
    recall = recall_score(Y_test, Y_Pred.astype(int),average="micro")

    accuracy_vals.append(accuracy)
    recall_vals.append(recall)
    precision_vals.append(precision)

    print(conf_matrixs)
    print("Accuracy for fold " + str(fold_count) + ": " + str(round(accuracy, 2))+"%")
    print("Precision for fold " + str(fold_count) + ": " + str(round(precision * 10, 2)) + "%")
    print("Recall for fold " + str(fold_count) + ": " + str(round(recall* 10, 2)) + "%")

    fold_count += 1


"""
TODO: iterating over each fold, fit a decision tree classifier on the training set.
Then predict the class labels for the test set and compute the confusion matrix
using predicted labels and ground truth values. Print the accuracy, precision and recall
for each fold.
"""
# TODO: calculate and print the average accuracy, precision and recall values over all 10 folds
avg_accur = 0
for val in accuracy_vals:
    avg_accur += val
avg_accur/=10;

avg_pre = 0
for val in precision_vals:
    avg_pre += val
avg_pre/=10;

avg_rec = 0
for val in recall_vals:
    avg_rec += val
avg_rec/=10;

print("Average Accuracy", round(avg_accur, 2), "%")
print("Average Precision", round(avg_pre, 2), "%")
print("Average Recall", round(avg_rec, 2), "%")
# TODO: train the decision tree classifier on entire dataset
entire_model = tree.fit(X, Y.astype(int))

# TODO: Save the decision tree visualization to disk - replace 'tree' with your decision tree and run the below line
export_graphviz(tree, out_file='tree.dot', feature_names = feature_names)

# TODO: Save the classifier to disk - replace 'tree' with your decision tree and run the below line
with open('classifier.pickle', 'wb') as f:
    pickle.dump(tree, f)