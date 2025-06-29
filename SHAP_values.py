#!/usr/bin/env python
# coding: utf-8

# Imports
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import os
import time
import gc
import math

from pandas import DataFrame
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import hdf5storage
from scipy.signal import butter, filtfilt, savgol_filter
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from scipy.fft import fft, fftfreq, fft2, rfft
from scipy.signal import iirnotch, freqz
from scipy.signal import spectrogram

import keras
import shap

# Luca code
def import_mat(fn):
    data = hdf5storage.loadmat(fn)
    raw = data["data"]["raw"][0][0]
    epoched = data["data"]["epoched"][0][0]
    event_times = data["data"]["event_times"][0][0]
    event_names = data["data"]["event_names"][0][0]
    param_dict = {}
    param_dict["hand"] = data["data"]["hand"][0][0][0]
    param_dict["spikes"] = data["data"]["spikes"][0][0][0][0][0]
    param_dict["aligned_to"] = data["data"]["alignedTo"][0][0][0][0][0]
    param_dict["angle"] = data["data"]["angle"][0][0][0]
    param_dict["event_times"] = event_times
    param_dict["event_names"] = event_names
    return raw, epoched, param_dict

def butter_bandpass(lowcut, highcut, fs, order=2):
    """
    Compute the filter coefficients for a Butterworth bandpass filter.
    """
    # Compute the Nyquist frequency
    nyq = 0.5 * fs
    # Compute the low and high frequencies
    low = lowcut / nyq
    high = highcut / nyq
    # Compute the filter coefficients
    b, a = butter(order, [low, high], btype="band")
    # Return the filter coefficients
    return b, a


def bandpass_filter(lfp, fs, lowcut, highcut):
    """
    Apply a bandpass filter to the LFP signal.
    """
    # Compute the filter coefficients
    b, a = butter_bandpass(lowcut, highcut, fs)
    # Apply the filter
    lfp_filtered = filtfilt(b, a, lfp, axis=0)
    # Return the filtered LFP signal
    return lfp_filtered

def compute_eeg_spectrograms(data, fs=2000, nperseg=200, noverlap=100):
    """
    Compute spectrograms for each neuron in EEG data.

    Parameters:
    - data: ndarray of shape (256, 508, num_trials, 5)
    - fs: Sampling frequency
    - nperseg: Length of each FFT segment
    - noverlap: Overlap between segments

    Returns:
    - spectrograms: ndarray of shape (num_total_trials, 256, n_freq_bins, n_time_windows)
    """
    n_neurons, n_time, n_trials, n_classes = data.shape
    all_specs = []

    for cls in range(n_classes):
        for trial in range(n_trials):
            trial_specs = []
            for neuron in range(n_neurons):
                signal = data[neuron, :, trial, cls]
                f, t, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
                trial_specs.append(Sxx)  # Sxx shape: (n_freq_bins, n_time_windows)
            trial_specs = np.stack(trial_specs, axis=0)  # Shape: (256, n_freq_bins, n_time_windows)
            all_specs.append(trial_specs)

    all_specs = np.stack(all_specs, axis=0)  # Shape: (n_total_trials, 256, n_freq_bins, n_time_windows)
    return f, all_specs.transpose(0, 2, 3, 1)

def clean_spectograms(spectograms, f, remove_freqs, fs, window_size, upper_bound):
    cleaned_spectograms = []
    lower_bound = int(math.ceil(fs / window_size / 10.0) * 10)
    lower_bound_indx = np.where(f >= lower_bound)[0][0]
    upper_bound_indx = np.where(f <= upper_bound)[0][-1]
    mask = (f >= remove_freqs-2) & (f <= remove_freqs+2)
    harmonics = np.arange(remove_freqs, f[-1], remove_freqs)

    for trial in spectograms:
        cleaned_spectogram_per_trial = []
        for channel in trial:
            cleaned_spectogram = channel
            for h in harmonics:
                mask = (f >= h-2) & (f <= h+2)
                cleaned_spectogram[mask, :] = 0
            cleaned_spectogram = cleaned_spectogram[lower_bound_indx:upper_bound_indx]
            cleaned_spectogram_per_trial.append(cleaned_spectogram)
        cleaned_spectograms.append(cleaned_spectogram_per_trial)
    f = f[lower_bound_indx:upper_bound_indx]
    return np.array(cleaned_spectograms), f

# Data load pipeline

# Load data
print('Loading 1...')
_, epoched1, _ = import_mat('1_data_left_0.mat')
print('Loading 2...')
_, epoched2, _ = import_mat('2_data_left_45.mat')
print('Loading 3...')
_, epoched3, _ = import_mat('3_data_left_90.mat')
print('Loading 4...')
_, epoched4, _ = import_mat('4_data_left_135.mat')
print('Loading 5...')
_, epoched5, _ = import_mat('5_data_right_0.mat')
print('Loading 6...')
_, epoched6, _ = import_mat('6_data_right_45.mat')
print('Loading 7...')
_, epoched7, _ = import_mat('7_data_right_90.mat')
print('Loading 8...')
_, epoched8, _ = import_mat('8_data_right_135.mat')

epoched_left = np.concatenate((epoched1, epoched2, epoched3, epoched4), axis=2)
epoched_right = np.concatenate((epoched5, epoched6, epoched7, epoched8), axis=2)
left_trials = epoched1.shape[2] + epoched2.shape[2] + epoched3.shape[2] + epoched4.shape[2]
right_trials = epoched5.shape[2] + epoched6.shape[2] + epoched7.shape[2] + epoched8.shape[2]

# Delete individual datasets to free memory
del epoched1, epoched2, epoched3, epoched4, epoched5, epoched6, epoched7, epoched8
del _

# Force garbage collection
gc.collect()

# Create spectrograms
print('Making spectrograms...')
fl, specs_left = compute_eeg_spectrograms(epoched_left)
_, specs_right = compute_eeg_spectrograms(epoched_right)

# Create labels
y_left = np.repeat(np.arange(0, 5), epoched_left.shape[2])
zeros = np.zeros(epoched_right.shape[2])
y_part_right = np.repeat(np.arange(5, 9), epoched_right.shape[2])
y_right = np.concatenate([zeros, y_part_right])
y_total = np.concatenate((y_left, y_right), axis=0)

# Clean spectrograms
print('Cleaning data...')
X_total = np.concatenate((specs_left, specs_right), axis=0)
X_total = X_total.transpose(0, 3, 1, 2)
X_final, f = clean_spectograms(X_total, fl, remove_freqs=60, fs=2000, window_size=200, upper_bound=500)
X_final = X_final.transpose(0, 2, 3, 1)

# Remove top channels (list pulled by Luca's code)
channels_to_remove = [0, 1, 3, 5, 9, 30, 52, 63, 81, 88, 92]
channels_to_keep = [i for i in range(X_final.shape[-1]) if i not in channels_to_remove]
X_final = X_final[:, :, :, channels_to_keep]

X_train, X_test, y_train, y_test = train_test_split(X_final, y_total, test_size=0.2, random_state=42)

del epoched_left, epoched_right, specs_left, specs_right, X_total, X_final, y_total, y_left, y_right
gc.collect()
print('Data loaded.')

# Load all models
print('Loading models...')
cnn_lstm_832 = keras.models.load_model("new_cnn_lstm_cleaned_832.keras")

parallel_822 = keras.models.load_model("new_parallel_cleaned_822.keras")

valid_models = ['parallel', 'cnn-lstm']

# Get inputs to determine training
while True:
    model_use = input(f"Enter a region name {valid_models}: ").strip()
    if model_use in valid_models:
        print(f"Getting SHAP values for model type: {model_use}")
        break
    else:
        print(f"Invalid input. Please choose from {valid_models}.")
train_size = int(input("Training samples: "))
nsamples = int(input("Number of samples for SHAP: "))
print(f"Taking a training subset of size {train_size} and using {nsamples} samples to create SHAP values...")

# Reshape dimensions
input_shape = (49, 4, 245)
flatten_shape = np.prod(input_shape)

# Wrap model.predict to reshape incoming flat input
def lstm_predict(X_flat):
    X = X_flat.reshape((-1, *input_shape))  # Reshape to (batch, 49, 4, 245)
    return cnn_lstm_832.predict(X)

def parallel_predict(X_flat):
    X = X_flat.reshape((-1, *input_shape))  # Reshape to (batch, 49, 4, 245)
    return parallel_822.predict(X)

# Background dataset (small subset of training data)
X_background = X_train[np.random.choice(X_train.shape[0], train_size, replace=False)]
X_background_flat = X_background.reshape((X_background.shape[0], -1))

# Test data to explain
X_test_subset = X_test[:10]  # First 10 samples
X_test_flat = X_test_subset.reshape((X_test_subset.shape[0], -1))

if model_use=="parallel":
    # Initialize KernelExplainer
    parallel_explainer = shap.KernelExplainer(parallel_predict, X_background_flat)
    # Compute SHAP values (this is slow for large batches)
    shap_values = parallel_explainer.shap_values(X_test_flat, nsamples=nsamples)
else:
    lstm_explainer = shap.KernelExplainer(lstm_predict, X_background_flat)
    shap_values = lstm_explainer.shap_values(X_test_flat, nsamples=nsamples) 

# Take absolute value
shap_abs = np.abs(shap_values[0])  # shape (48020, 9)

# Aggregate across classes — average importance per feature
shap_mean_across_classes = shap_abs.mean(axis=1)  # shape (48020,)

# Reshape back to original input shape
shap_mean_reshaped = shap_mean_across_classes.reshape((49, 4, 245))

importance_per_channel = shap_mean_reshaped.mean(axis=(0, 1))
top_channels = np.argsort(-importance_per_channel)[:10]  # Indices of top channels
top_importance_values = importance_per_channel[top_channels]  # Corresponding SHAP magnitudes

original_channel_index = []
for i in top_channels:
    original_channel_index.append(channels_to_keep[i])

print("CNN-LSTM top channels (indices):", top_channels)
print("Top channels (original):", original_channel_index)
print("Their importance scores:", top_importance_values)

# Plot importances

regions = {
    'l_PMv': (0, 55),
    'l_PMd': (56, 84),
    'r_PMd': (85, 148),
    'r_PMv': (149, 212),
    'l_M1': (213, 244)
}

region_colors = {
    'l_PMv': '#E69F00',  # orange
    'l_PMd': '#56B4E9',  # sky blue
    'r_PMd': '#009E73',  # green
    'r_PMv': '#F0E442',  # yellow
    'l_M1':  '#CC79A7',  # redish purple
}

def get_region_for_channel(channel_idx):
    for region, (start, end) in regions.items():
        if start <= channel_idx <= end:
            return region

channel_indices = np.arange(len(importance_per_channel))
colors = [region_colors[get_region_for_channel(i)] for i in channel_indices]

plt.figure(figsize=(12, 4))
plt.bar(channel_indices, importance_per_channel, color=colors)
plt.xlabel("Channel Index")
plt.ylabel("Mean |SHAP Value|")
handles = [plt.Rectangle((0,0),1,1, color=region_colors[label]) for label in regions.keys()]
plt.legend(handles, regions.keys(), title="Regions")
if model_use=="parallel":
    plt.title("Parallel SHAP KernelExplainer Channel Importance")
    plt.savefig("parallel_ke_channels.pdf", format='pdf')
else:
    plt.title("CNN-LSTM SHAP KernelExplainer Channel Importance")
    plt.savefig("cnn_lstm_ke_channels.pdf", format='pdf')    
plt.show()






