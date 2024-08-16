# -*- coding: utf-8 -*-
"""
Created on 18-01-2024

@author: Bence Many, Richard Thulstrup
"""
import numpy as np
import matplotlib.pyplot as plt
import doppler_lib as dl
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import hilbert, detrend, firwin, filtfilt
import math
import time
from doppler_lib import *
from tqdm import tqdm


plotting = True

# Read the data file
filename = dl.find_file()
data, x_pos_range, time_range = dl.read_csv(filename, alfa, gain_comp=True)

plt.close('all')




pos = data[0]
segment = pos[0]
segment_images = []

# Define progress bar
bar_format = "{l_bar}{bar}, Time remaining: {remaining} s"

for segment in tqdm(pos, desc="Processing data",  bar_format=bar_format):
        
    velocity_profiles = []
    
    # Analysis of segment
    for line in segment:
        
        velocity = dl.doppler_signal_processing(line)
        
        # Averaging velocity lines
        velocity_lines = []
        for i in range(number_of_pulses-3):
            vel_i = velocity[i * bin_size : (i + 1) * bin_size]
            velocity_lines.append(vel_i)
            
        average_velocity = np.sum(velocity_lines, axis=0) / len(velocity_lines)
        velocity_profiles.append(average_velocity)
        
        
    v_max = 0.05
    transposed = [list(row) for row in zip(*velocity_profiles)]
    segment_images.append(transposed)


# Display dataset parameters
print()
print("New segment")
print("Amount of positions: ", len(data))
print("Amount of segments in pos: ", len(pos))
print("Amount of lines in segment: ", len(segment))
print("Length of one line: ", len(line))

# Stack segment images vertically
full_image = np.vstack(segment_images)

# Define custom colormap
colors = [(0, 0, 1), (0, 0, 0), (1, 0, 0)]  # Red, Black, Blue
n_bins = 100
cmap_name = "doppler_cmap"
doppler_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    
#Display Doppler image
y_scale_factor = 1540 / 60e6 / 2 * 1e3
v_max = 0.01
plt.figure()
plt.imshow(full_image, cmap=doppler_cmap, vmin=-v_max, vmax=v_max,
            extent=(0, 10, len(full_image) * y_scale_factor, 0))
plt.title("Doppler Color Flow Image")
plt.ylabel("Depth (mm)")
plt.xlabel("Pulse number")
plt.colorbar(label='Velocity (m/s)')





# ******************************************************* OLD CODE *****************************************************
# start_time = time.time()

# image = np.empty((0, len(data[1])))

# for depth in data[1:]:
#     depth_image = []
#     for line in depth:
#         filtered = dl.band_pass_filter(line)
#         pulses = []
#         cancelled = []
        
#         #Extracting pulses from RF lines
#         for i in range(number_of_pulses-1):
#             pulse_i = filtered[i * pulse_length : (i + 1) * pulse_length]
#             # hann_window = np.hanning(len(pulse_i))
#             # pulse_i = pulse_i * hann_window
#             pulses.append(pulse_i)
        
#         #Cleaning the dataset
#         limit = np.median([np.max(signal) for signal in pulses]) * 3
#         pulses_cleaned = [pulse for pulse in pulses if np.max(pulse) <= limit]
        
#         #Echo cancellation
#         for i in range(len(pulses_cleaned)-1):
#             p1_norm = pulses_cleaned[i+1] / np.max(np.abs(pulses_cleaned[i+1]))
#             scale_factor = np.dot(p1_norm, pulses_cleaned[i]) / np.dot(pulses_cleaned[i], pulses_cleaned[i])
#             p2_adjusted = scale_factor * pulses_cleaned[i]
#             diff_i = p1_norm - p2_adjusted
#             # signal_hilbert = hilbert(diff_i)   #Hilbert-transorm
#             # signal_envelope = np.abs(signal_hilbert)    #Calculating the envelope signal
#             cancelled.append(diff_i)
            
#         #Time-shift estimation
#         shifts = []
#         for i in range(len(cancelled)-1):
#             if dl.reliability_calculator(cancelled[i], cancelled[i+1]) > rel_limit:
#                 cross_corr_results = np.correlate(cancelled[i], cancelled[i+1], mode='same')
#                 peak = np.where(cross_corr_results == np.max(cross_corr_results))[0][0]
#                 shift = (pulse_length / 2 - peak)
#                 velocity = (c/2) * (shift/fs) * (prf/pulse_delay) * 100 / math.cos(math.radians(angle))
#                 shifts.append(velocity)
#             else:
#                 shifts.append(0)
            
            
#         depth_image.append(np.mean(shifts, axis = 0))
#     transposed = np.array(depth_image).T       #Rotating the image
#     image = np.vstack((image, transposed))

# #Average filtering
# filtered_image = dl.moving_avg_filter(image, window_size=2)

# #Locating the flow        
# max_flow = 0
# max_flow_location = None
# for i, sublist in enumerate(filtered_image):
#     flow = max(sublist, key=abs)
#     if abs(flow) > abs(max_flow):
#         max_flow = flow
#         max_flow_location = i
# print(f"\nThe highest flow rate was {round(max_flow, 2)} cm/s, detected at depth {round((min(depths) + max_flow_location) * 128 * 1.48 / 60 / 2, 2)} mm.")


# # Display Doppler Color Flow image
# v_max = round(abs(max_flow)) + 2
# dl.display_doppler_image(filtered_image, times, depths, v_max)
# plt.text(3.5, 1.5, f"Detected flow: {round(max_flow, 2)} cm/s \n Depth: {round((min(depths) + max_flow_location) * 128 * 1.48 / 60 / 2, 2)} mm", fontsize=12, color='red')


# print("\nProcessing finished.")
# end_time = time.time()
# elapsed_time = end_time - start_time
# print("Total processing time:", round(elapsed_time, 1), "seconds")







    