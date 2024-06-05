# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 13:16:16 2024

@author: Bence Many

Library for Doppler Imaging
"""

from tkinter import Tk
from tkinter.filedialog import askopenfilename
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.colors import LinearSegmentedColormap
import math
from scipy.signal import bessel, freqz, lfilter


conv_echo = 1.48 / 160  #Conversion from sample to distance (mm)
prf = 5000              #Pulse Repetition Rate = 5 kHz
f0 = 4e6                #Probe frequency = 4 MHz
fs = 80e6               #Sampling frequency = 80 MHz
c = 1540                #Speed of sound
angle = 45              #Angle of probe in degrees
v_max = c/4 * (prf/f0) * 100/math.cos(math.radians(angle))    #Max detectable velocity in cm/s
rel_limit = 0.0        #Lower limit for acceptable reliability index
number_of_segments = 2  #Number of segments in a sample
pulse_delay = 1        #Delay between pulses to compare
bin_size = 128
number_of_pulses = int(7168/bin_size)

def b_mode(frame):
    image = []
    for pos in frame:
        line = []
        for segment in pos:
            line += segment[:512]
        image.append(line)
    return np.array(image).T

def find_file():
        
    #Open file dialog to pick the desired BoM
    root = Tk()
    root.attributes('-topmost', True)
    root.iconify() 
    file_path = askopenfilename(title='Select M-mode data', parent=root)
    print("File selected: ", file_path)
    root.destroy()
    return file_path

def read_csv(file_name):
    data = []
    positions = []
    depths = []
    time = []
    
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        
        header = next(reader)  # Read the header
        if not header:
            return
        
        # Check if header matches expected format
        expected_header = ["HILOVAL", "STRATEG_IDX", "TIME", "DACVAL", "OFFSET"] + [f"D[{i}]" for i in range(7168)]
        if header != expected_header:
            raise ValueError("Unexpected header format")

        xpos = float(next(reader)[1])
        for row in reader:
            if float(row[1]) != xpos: 
                positions.append(data)
                time.append(float(row[2]))
                data = []
                xpos = float(row[1])
            data.append([float(val) for val in row[5:]])
            depths.append(xpos)
        positions.append(data)
    print("Data read successfully")
                    
    return positions, depths, time 

def reliability_calculator(signal1, signal2):
    sign1 = np.sign(signal1)
    sign2 = np.sign(signal2)
    corr_res = np.correlate(sign1, sign2, mode='same')
    corr_res /= len(sign1)
    rel_index = max(corr_res)
    return rel_index

def time_shift(signal_1, signal_2, plotting=False):
    
    # Perform cross-correlation for each segment
    segment_size = int(bin_size / number_of_segments)
    shifts_in_region = []
    for i in range(0, number_of_segments):
        segment1 = signal_1[i * segment_size : (i+1) * segment_size]
        segment2 = signal_2[i * segment_size : (i+1) * segment_size]

        # Calculate cross-correlation for the current segment
        cross_corr_results = np.correlate(segment1, segment2, mode='same')
        peak = np.where(cross_corr_results == np.max(cross_corr_results))[0][0]
        shift = (segment_size / 2 - peak)
        shifts_in_region.append(shift)
    if plotting:
        plt.figure()
        plt.plot(shifts_in_region)
    return max(shifts_in_region, key=abs)

def calculate_velocity(section, plotting=False):
    
    #Extracting pulses from segments
    pulses = []
    for i in range(0,number_of_pulses):
        bin_i = section[i * bin_size : (i + 1) * bin_size]
        pulses.append(bin_i)
    if plotting:
        figure()
        for i in range(4):
            plt.plot(pulses[i], label=f"{i}.pulse")
            plt.title(f"{i} pulses in a section")
            plt.legend()
    
    #Measure the time-shift betwen pulses   
    if plotting: figure()
    shifts = []
    diff_pulses = []
    for j in range(len(pulses)-pulse_delay):
        signal_1 = pulses[j]
        signal_2 = pulses[j+pulse_delay]
        
        #Normalize the pulses
        signal_1_norm = signal_1 / np.max(np.abs(signal_1))
        
        # Adjust the second pulse to minimize residual variance in the subtraction
        scale_factor = np.dot(signal_1_norm, signal_2) / np.dot(signal_2, signal_2)
        signal_2_adjusted = scale_factor * signal_2
        
        # Subtract the adjusted second pulse from the normalized first pulse
        diff_pulses.append(signal_1_norm - signal_2_adjusted)
    
        if reliability_calculator(signal_1, signal_2) > rel_limit:
            shifts.append(time_shift(signal_1, signal_2))
        else: shifts.append(0)
    
    #Remove outliers
    filtered_shifts = shifts
    
    if plotting:
        figure()
        plt.plot(filtered_shifts)
        plt.title("Shift between pulses in segment")
        
    if len(filtered_shifts) > 0: shift_avg = sum(filtered_shifts) / len(filtered_shifts)
    else: shift_avg = 0 
        
    #Estimate velocity based on time-shift
    velocity = (c/2) * (shift_avg/fs) * (prf/pulse_delay) * 100 / math.cos(math.radians(angle))
    return velocity, diff_pulses

def band_pass_filter(signal, f_low=3.5e6, f_high=5e6):
    
    # Design the Bessel bandpass filter
    order = 5   # Filter order (5th order Bessel)
    Wn_low = 2 * f_low / fs
    Wn_high = 2 * f_high / fs
    
    #Create the filter
    b, a = bessel(order, [Wn_low, Wn_high], btype='bandpass', analog=False)
    
    #Applying bandpass filter
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal

def signal_processing(data):
        
    data_filtered = []
    for line in data:
        segment = []
        for signal in line:
            
            #Apply bandpass filter
            filtered = band_pass_filter(signal)
            
            # Apply Hann window
            hann_window = np.hanning(len(filtered))
            windowed_signal = filtered * hann_window
            segment.append(windowed_signal)
        data_filtered.append(segment)
    return data_filtered

def display_doppler_image(image, x_range, y_range, v_max=5):
    
    x_scale_factor = 10e6
    y_scale_factor = 128 * 1.48 / 80 / 2
    
    # Define custom colormap
    colors = [(0, 0, 1), (0, 0, 0), (1, 0, 0)]  # Red, Black, Blue
    n_bins = 100
    cmap_name = "doppler_cmap"
    doppler_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    
    #Display Doppler Color FLow image
    plt.figure()
    display = np.array(image)
    plt.imshow(display, cmap=doppler_cmap, vmin=-v_max, vmax=v_max, 
               extent=(min(x_range) / x_scale_factor, max(x_range) / x_scale_factor, max(y_range) * y_scale_factor, min(y_range) * y_scale_factor))
    plt.title("Doppler Color Flow Image")
    plt.ylabel("Depth segment (mm)")
    plt.xlabel("Time (s)")
    plt.colorbar(label='Velocity (cm/s)')
    
def median_filter(image, kernel_size=3):
    padded_image = np.pad(image, pad_width=kernel_size//2, mode='constant', constant_values=0)
    filtered_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            filtered_image[i, j] = np.median(padded_image[i:i+kernel_size, j:j+kernel_size])
    return filtered_image

def moving_avg_filter(image, window_size=10):
    filtered_image = []
    kernel = np.ones(window_size) / window_size
    for line in image:
        filtered_line = np.convolve(line, kernel, mode='same')
        filtered_image.append(filtered_line)    
    return filtered_image


    

    
