# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 12:43:04 2024

@author: Bence Many
"""

import doppler_lib as dl
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

sampling_freq = 60       #Sampling frequency

file_name = dl.find_file()
x_range, data, times = dl.read_b_mode_csv(file_name)

signals = []
for pos in data:
    line = pos[0]
    filtered = dl.band_pass_filter(line)
    signal_hilbert = hilbert(filtered)   #Hilbert-transorm
    signal_envelope = np.abs(signal_hilbert)    #Calculating the envelope signal
    signals.append(signal_envelope)
image = np.array(signals).T       #Rotating the image

#Logarithmic mapping
c = 255 / np.log(1 + np.max(image))
log_image = c * (np.log(image + 1))

# filtered_image = dl.median_filter(log_image, kernel_size=10)

#Display B-mode image
x_scale_factor = 10e6
y_scale_factor = 1.48 / sampling_freq / 2
min_pos = min(x_range)
max_pos = max(x_range)
min_depth = 0
max_depth = len(log_image) * y_scale_factor

#Displaying the B-mode image
plt.figure()
display = np.array(log_image)
plt.imshow(display, cmap="gray",
           extent=(min_pos, max_pos, max_depth, min_depth))
plt.title("Logarithmic B-mode image")
plt.ylabel("Depth (mm)")
plt.xlabel("X-position (mm)")