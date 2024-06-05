# -*- coding: utf-8 -*-
"""
Created on 18-01-2024

@author: Bence Many, Richard Thulstrup
"""
import numpy as np
import matplotlib.pyplot as plt
import doppler_lib as dl
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import hilbert
import math


fs = 80e6
pulse_length = 128
number_of_pulses = int(7168/pulse_length)
rel_limit = 0.85
c = 1540                #Speed of sound
prf = 5000              #Pulse Repetition Rate = 5 kHz
angle = 45              #Angle of probe in degrees
pulse_delay = 1        #Delay between pulses to compare

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
        
        #Extracting pulses from RF lines
        for i in range(number_of_pulses-1):
            pulse_i = filtered[i * pulse_length : (i + 1) * pulse_length]
            hann_window = np.hanning(len(pulse_i))
            pulse_i = pulse_i * hann_window
            pulses.append(pulse_i)
        
        #Cleaning the dataset
        limit = np.median([np.max(signal) for signal in pulses]) * 3
        pulses_cleaned = [pulse for pulse in pulses if np.max(pulse) <= limit]
        
        #Echo cancellation
        for i in range(len(pulses_cleaned)-1):
            p1_norm = pulses_cleaned[i+1] / np.max(np.abs(pulses_cleaned[i+1]))
            scale_factor = np.dot(p1_norm, pulses_cleaned[i]) / np.dot(pulses_cleaned[i], pulses_cleaned[i])
            p2_adjusted = scale_factor * pulses_cleaned[i]
            diff_i = p1_norm - p2_adjusted
            # signal_hilbert = hilbert(diff_i)   #Hilbert-transorm
            # signal_envelope = np.abs(signal_hilbert)    #Calculating the envelope signal
            cancelled.append(diff_i)
            
        #Time-shift estimation
        shifts = []
        for i in range(len(cancelled)-1):
            if dl.reliability_calculator(cancelled[i], cancelled[i+1]) > rel_limit:
                cross_corr_results = np.correlate(cancelled[i], cancelled[i+1], mode='same')
                peak = np.where(cross_corr_results == np.max(cross_corr_results))[0][0]
                shift = (pulse_length / 2 - peak)
                velocity = (c/2) * (shift/fs) * (prf/pulse_delay) * 100 / math.cos(math.radians(angle))
                shifts.append(velocity)
            else:
                shifts.append(0)
            
            
        depth_image.append(np.mean(shifts, axis = 0))
    transposed = np.array(depth_image).T       #Rotating the image
    image = np.vstack((image, transposed))

#Average filtering
filtered_image = dl.moving_avg_filter(image, window_size=15)


#Locating the flow        
max_avg = float('-inf')
max_avg_location = None
for i, sublist in enumerate(filtered_image):
    avg = max(abs(sublist))
    if avg > max_avg:
        max_avg = avg
        max_avg_location = i
print(f"The highest flow rate was {round(max_avg, 2)} cm/s, detected at depth {round((min(depths) + max_avg_location) * 128 * 1.48 / 80 / 2, 2)} mm.")


# Display Doppler Color Flow image
v_max = round(max_avg) + 2
dl.display_doppler_image(filtered_image, time, depths, v_max)

# Display Flow signal
x_scale = np.linspace(min(time)/10e6, max(time)/10e6, len(filtered_image[max_avg_location]))
plt.figure()
plt.plot(x_scale, filtered_image[max_avg_location])
plt.title("Detected flow pattern")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (cm/s)")








    