# -*- coding: utf-8 -*-
"""
Created on 18-01-2024

@author: Richard, Bence
"""
import numpy as np
import matplotlib.pyplot as plt
import doppler_lib as dl
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import hilbert


fs = 80e6
pulse_length = 128
number_of_pulses = int(7168/pulse_length)


# Read the data file
filename = dl.find_file()
data, depths, time = dl.read_csv(filename)

image = np.empty((0, len(data[1])))
for depth in data[1:]:
    depth_image = []
    for line in depth:
        filtered = dl.band_pass_filter(line)
        pulses = []
        cancelled = []
        for i in range(number_of_pulses):
            pulse_i = filtered[i * pulse_length : (i + 1) * pulse_length]
            hann_window = np.hanning(len(pulse_i))
            pulse_i = pulse_i * hann_window
            pulses.append(pulse_i)
        for i in range(len(pulses)-1):
            p1_norm = pulses[i+1] / np.max(np.abs(pulses[i+1]))
            scale_factor = np.dot(p1_norm, pulses[i]) / np.dot(pulses[i], pulses[i])
            p2_adjusted = scale_factor * pulses[i]
            diff_i = p1_norm - p2_adjusted
            signal_hilbert = hilbert(diff_i)   #Hilbert-transorm
            signal_envelope = np.abs(signal_hilbert)    #Calculating the envelope signal
            cancelled.append(signal_envelope)
        depth_image.append(np.mean(cancelled, axis = 0))
    transposed = np.array(depth_image).T       #Rotating the image
    image = np.vstack((image, transposed))

c = 255 / np.log(1 + np.max(image))
log_image = c * (np.log(image + 1))    #Logarithmic transformation of the image

fig = plt.figure()
plt.imshow(log_image, cmap="gray", extent=(min(time) / 10e6, max(time) / 10e6, max(depths) * 128 * 1.48 / 80 / 2, min(depths) * 128 * 1.48 / 80 / 2))  
plt.title("Stationary echo cancelled pulses")
plt.xlabel("Time (s)")
plt.ylabel("Depth (mm)")

    