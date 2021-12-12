#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 14:19:34 2021

Class Code for 10/21/21
Adding onto code from 10/19/21

@author: annajanebrown
"""

# Import packages
import numpy as np
from matplotlib import pyplot as plt

# %% Create synthetic data
T = 2 # duration of signal in seconds
fs = 100 # sampling frequency in Hz
t = np.arange(0,T,1/fs) # time of each sample in seconds

# Create two sine waves at different frequencies
freq_A = 1 # frequency of first signal in Hz
freq_B = 10 # frequency of second signal in Hz
amplitude_A = 1 # amplitude of first signal
amplitude_B = 1 # amplitude of second signal

sine_A = amplitude_A * np.sin(2*np.pi * t * freq_A)
sine_B = amplitude_B * np.sin(2*np.pi * t * freq_B)

# combine the two since waves
signal = sine_A + sine_B

# What happens if we change our signal to be one that's not so simple?
# Let's creat a signal that simulates brain activity stopping after a certain length of time
#signal[t>T/2]=0

# Here's another more complex signal
signal[:]=0
signal[t==1]=1

# %% Plot the synthetic data
# Set up a figure

plt.figure(363)
plt.clf()

# Plot first sine wave
plt.subplot(3,1,1)
plt.plot(t, sine_A)
# Annotate plot
plt.xlabel('time (s)')
plt.ylabel('voltage (uV)')
plt.grid()

# Plot second sine wave
plt.subplot(3,1,2)
plt.plot(t, sine_B)
# Annotate plot
plt.xlabel('time (s)')
plt.ylabel('voltage (uV)')
plt.grid()

# Plot combined signal
plt.subplot(3,1,3)
plt.plot(t, signal)
# Annotate plot
plt.xlabel('time (s)')
plt.ylabel('voltage (uV)')
plt.grid()

plt.tight_layout()

# %% Calculate Fourier Transform of signal

# Take FFT of signal 
signal_fft = np.fft.rfft(signal)

# Get spectrum (signal power)
signal_power = signal_fft *signal_fft.conj()
# normalize spectrum
signal_power_norm = signal_power/np.max(signal_power)
# convert to decibels
signal_power_db = 10*np.log10(signal_power_norm)

# Get frequencies of FFT
signal_fft_freqs = np.fft.rfftfreq(len(signal)) * fs

# Plot the power spectrum in dB
plt.figure(453)
plt.clf()
plt.plot(signal_fft_freqs, signal_power_db)

# Annotate the plot
plt.title('signal power spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (dB)')
plt.tight_layout()

# Everything from here down is new!
# %% Create a band-stop filter to remove the 10 Hz signal

# create band-stop filter to remove
filter_frequncy_response = np.ones(len(signal_fft_freqs))
filter_frequncy_response[signal_fft_freqs==10] = 0 

# Plot frequency response of filter
plt.figure(234)
plt.clf()
plt.plot(signal_fft_freqs, filter_frequncy_response)
plt.title('gain of filer')
plt.xlabel('Frequency (Hz)')
plt.ylabel('filter gain')

# Multiply signal FFT by frequncy response
filtered_fft = signal_fft * filter_frequncy_response

# Take inverse FFT of filtered signal
filtered_signal = np.fft.irfft(filtered_fft)

# %% Plot raw and filtered signal

plt.figure(534)
plt.clf()
plt.plot(t, signal)
plt.plot(t, filtered_signal)
# Annotate plot
plt.xlabel('time (s)')
plt.ylabel('voltage (uV)')
plt.grid()

