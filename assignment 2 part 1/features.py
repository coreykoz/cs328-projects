# -*- coding: utf-8 -*-
"""
This file is used for extracting features over windows of tri-axial accelerometer 
data. We recommend using helper functions like _compute_mean_features(window) to 
extract individual features.

As a side note, the underscore at the beginning of a function is a Python 
convention indicating that the function has private access (although in reality 
it is still publicly accessible).

"""

import numpy as np
from scipy.signal import find_peaks
from math import log, atan, pi, sqrt
from statistics import variance
import warnings

def _compute_mean_features(window):
    """
    Computes the mean x, y and z acceleration over the given window. 
    """
    return np.mean(window, axis=0)

def _compute_peak_features(window):
    steps_indicies, ___ = find_peaks(window, height=16)
    return len(steps_indicies)

def _compute_rfft_features(window):
    warnings.filterwarnings("ignore")
    result = np.fft.rfft(window, axis=0).astype(float)
    return result

def _compute_peak_rfft_features(window):
    return max(_compute_rfft_features(window))

def _compute_acceleration_features(window):
    time_period = len(window)
    accel = 0
    for count in range(time_period-1):
        initial_val = window[count]
        final_val = window[count+1]
        accel += (final_val-initial_val)
    return accel/time_period

def _compute_entropy_features(window):
    probability_dist = np.histogram(window, density=True)[0]
    entropy_val = 0

    for value in probability_dist:
        if value != 0:
            entropy_val += (-1*value) * log(value, 2)
    return entropy_val

def _compute_varaince_features(window):
    return variance(window)

def _compute_pitch_features(window):
    all_pitches = []

    for row in window:
        x = (row[0])
        y = (row[1]**2)
        z = (row[2]**2)

        pitch = atan(x/sqrt(y + z)) * (180/pi)
        all_pitches.append(pitch)

    sum1 = 0
    for item in all_pitches:
        sum1 += item
    return sum1/len(all_pitches)


# TODO: define functions to compute more features

def extract_features(window):
    """
    Here is where you will extract your features from the data over 
    the given window. We have given you an example of computing 
    the mean and appending it to the feature vector.
    
    """

    x = []
    feature_names = []

    magnitude = []
    for row in window:
        magnitude.append(np.sqrt(row[0] ** 2 + row[1] ** 2 + row[2] ** 2))

    # Catergory 1: Stat Features
    x.append(_compute_mean_features(window[:, 0]))
    feature_names.append("x_mean")
    '''
    x.append(_compute_mean_features(window[:, 1]))
    feature_names.append("y_mean")
    '''
    x.append(_compute_mean_features(window[:, 2]))
    feature_names.append("z_mean")

    x.append(_compute_varaince_features(magnitude))
    feature_names.append("variance of magnitude")
    '''
    x.append(_compute_varaince_features(window[:, 0]))
    feature_names.append("variance of x")
    
    x.append(_compute_varaince_features(window[:, 1]))
    feature_names.append("variance of y")

    x.append(_compute_varaince_features(window[:, 2]))
    feature_names.append("variance of z")
    '''
    # Catergory 2: FFT Features
    x.append(_compute_peak_rfft_features(magnitude))
    feature_names.append("max rfft mag")

    x.append(_compute_peak_rfft_features(window[:, 0]))
    feature_names.append("max rfft x")

    x.append(_compute_peak_rfft_features(window[:, 1]))
    feature_names.append("max rfft y")

    x.append(_compute_peak_rfft_features(window[:, 2]))
    feature_names.append("max rfft z")

    # Catergory 3: Other Features
    x.append(_compute_entropy_features(window))
    feature_names.append("entropy of window")

    x.append(_compute_pitch_features(window))
    feature_names.append("pitch")
    '''
    x.append(_compute_acceleration_features(window[:, 0]))
    feature_names.append("acceleration? x")

    x.append(_compute_acceleration_features(window[:, 1]))
    feature_names.append("acceleration? y")

    x.append(_compute_acceleration_features(window[:, 2]))
    feature_names.append("acceleration? z")

    x.append(_compute_acceleration_features(magnitude))
    feature_names.append("acceleration? mag")
    '''
    # Catergory 4: Peak Features
    x.append(_compute_peak_features(magnitude))
    feature_names.append("peaks on magnitude")

    x.append(_compute_peak_features(window[:, 0]))
    feature_names.append("peaks on x")

    x.append(_compute_peak_features(window[:, 1]))
    feature_names.append("peaks on y")

    x.append(_compute_peak_features(window[:, 2]))
    feature_names.append("peaks on z")


    #feature_vector = np.concatenate(x, axis=0) # convert the list of features to a single 1-dimensional vector
    feature_vector = x
    return feature_names, feature_vector