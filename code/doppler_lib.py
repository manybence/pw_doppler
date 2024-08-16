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
from scipy.signal import bessel, freqz, lfilter, hilbert, detrend, firwin, filtfilt


conv_echo = 1.48 / 160  #Conversion from sample to distance (mm)
prf = 5000              #Pulse Repetition Rate = 5 kHz
f0 = 4e6                #Probe frequency = 4 MHz
fs = 60e6               #Sampling frequency = 80 MHz
c = 1540                #Speed of sound
angle = 45              #Angle of probe in degrees
v_max = c/4 * (prf/f0) * 100/math.cos(math.radians(angle))    #Max detectable velocity in cm/s
rel_limit = 0.0        #Lower limit for acceptable reliability index
number_of_segments = 2  #Number of segments in a sample
pulse_delay = 1        #Delay between pulses to compare
bin_size = 128
number_of_pulses = int(7168/bin_size)

#FPGA constants
prescale_count = 64

#Transducer parameters
attn_probe = f0 / 1e6
offset_d = 100 * 0.5 * c * prescale_count / fs      # cm
alfa = 10 ** ((offset_d * attn_probe) / 20)        # depth compensation as pr offset

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
    file_path = askopenfilename(title='Select Doppler data', parent=root, filetypes=[("CSV files", "*.csv")])
    print("File selected: ", file_path)
    root.destroy()
    return file_path

def read_b_mode_csv(file_name):
    data = []
    positions = []
    time = []
    x_pos_range = []
    
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        
        header = next(reader)  # Read the header
        if not header:
            return
        
        # Check if header matches expected format
        expected_header = ["XPOS", "STRATEG_IDX", "TIME", "DACVAL", "OFFSET"] + [f"D[{i}]" for i in range(7168)]
        if header != expected_header:
            raise ValueError("Unexpected header format")

        xpos = float(next(reader)[0])
        for row in reader:
            if float(row[0]) != xpos: 
                positions.append(data)
                time.append(float(row[2]))
                data = []
                xpos = float(row[0])
                x_pos_range.append(xpos)
            data.append([float(val) for val in row[5:]])
        positions.append(data)
    print("Data read successfully")
                    
    return x_pos_range, positions, time 

def read_csv(file_name, alfa, gain_comp=True):
    data = []
    position = []
    segment = []
    line = []
    time_range = []
    x_pos_range = []
    
    print("Reading data file...")
    
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        
        header = next(reader)  # Read the header
        if not header:
            return
        
        # Check if header matches expected format
        expected_header = ["XPOS", "STRATEG_IDX", "TIME", "DACVAL", "OFFSET"] + [f"D[{i}]" for i in range(7168)]
        expected_header2 = ["HILOVAL", "STRATEG_IDX", "TIME", "DACVAL", "OFFSET"] + [f"D[{i}]" for i in range(7168)]
        if (header != expected_header) and (header != expected_header2):
            raise ValueError("Unexpected header format")

        #Read first line
        first_line = next(reader)
        xpos, segment_id = int(first_line[0]), int(first_line[1])
        time_range.append(int(first_line[2]))
        line = [int(val) for val in first_line]
        if gain_comp: segment.append(gain_compensation(line, alfa)[5:])
        else: segment.append(line[5:])
        
        # Read data lines. Collect per position per segment
        for row in reader:
            
            # Start of new segment
            if int(row[1]) != segment_id: 
                position.append(segment)
                segment = []
                segment_id = int(row[1])
                
            # Start of new position
            if int(row[0]) != xpos: 
                data.append(position)
                position = []
                xpos = int(row[0])
                x_pos_range.append(xpos)
                
            # Start of new line
            time_range.append(int(row[2]))
            line = [int(val) for val in row]
            if gain_comp: segment.append(gain_compensation(line, alfa)[5:])
            else: segment.append(line[5:])
            
        position.append(segment)
        data.append(position)
        
        
    print("Data read successfully")
                    
    return data, x_pos_range, time_range 

def gain_compensation(row, alfa):
    row_new = np.array(row.copy(), dtype=float)
    current_dacval = row[3]
    current_offset = row[4]
    restore_factor = 10 ** ((current_dacval - 437) / 409)
    a = alfa * current_offset
    row_new[5:] = [val * restore_factor * a for val in row[5:]]
    return row_new

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

def doppler_signal_processing(line, plotting=False):
    
    # Extracting pulses from RF lines
    pulses = []
    for i in range(number_of_pulses-1):
        pulse_i = line[i * bin_size : (i + 1) * bin_size]
        pulses.append(pulse_i)
        
    
    # Create a matched filter using the median pulse    
    pulses_filtered = matched_filtering(pulses)
        

    # Echo cancellation
    cancelled_pulses = []
    for i in range(len(pulses_filtered)-1):
        diff_i = pulses_filtered[i+1] - pulses_filtered[i]
        cancelled_pulses.append(diff_i)
    cancelled_line = np.concatenate(cancelled_pulses)            
        
        
    # Band-pass filter
    filtered = band_pass_filter(cancelled_line)           
        
    
    # IQ demodulation
    t = np.arange(len(filtered)) / fs
    I = filtered * np.cos(2 * np.pi * f0 * t)
    Q = filtered * np.sin(2 * np.pi * f0 * t)
    iq_data = I + 1j * Q
        
    # IQ detrending
    detrended_iq_data = detrend(iq_data.real) + 1j * detrend(iq_data.imag)
    
    # Apply a low-pass FIR filter with a cutoff frequency of 15 kHz
    coefficients = firwin(100 + 1, 15e3 / (fs / 2))
    filtered_data_real = filtfilt(coefficients, 1.0, np.real(detrended_iq_data))
    filtered_data_imag = filtfilt(coefficients, 1.0, np.imag(detrended_iq_data))
    filtered_data = filtered_data_real + 1j * filtered_data_imag

    
    # Doppler velocity calculation using I/Q autocorrelation
    IQ1 = filtered_data[:-1]  # All elements except the last one
    IQ2 = filtered_data[1:]   # All elements except the first one
    R1 = IQ1 * np.conj(IQ2)   # Element-wise multiplication with conjugate
    velocity = -c * prf / (4 * f0) * np.imag(np.log(R1)) / np.pi
    
    if plotting:
            
            # Plot the original pulses
            plt.figure()
            for p in pulses:
                plt.plot(p)
            plt.title("Original pulses")
            
            plt.figure()
            plt.title("Match filtered pulses")
            for p in pulses_filtered:
                plt.plot(p)
                
            plt.figure()
            plt.title("Echo cancelled")
            plt.plot(cancelled_line)
            
            plt.figure()
            plt.title("Band-pass filtered")
            plt.plot(filtered)
            
            plt.figure()
            plt.title("Detrended IQ data")
            plt.plot(detrended_iq_data)
            
            plt.figure()
            plt.title("LP filtered IQ data")
            plt.plot(filtered_data)

    return velocity

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
    
def display_doppler_image_per_pos(image, v_max=5):
    
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
               extent=(0, len(image), len(image[0]) * y_scale_factor, 0))
    plt.title("Doppler Color Flow Image")
    plt.ylabel("Depth segment (mm)")
    plt.xlabel("Pulse number")
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

def matched_filtering(pulses):
    
    # Compute the median pulse
    median_pulse = np.median(pulses, axis=0)
    
    # Create a matched filter using the median pulse
    matched_filter = np.flip(median_pulse)
    
    # Apply the matched filter to each pulse
    filtered_pulses = []
    for p in pulses:
        filtered_pulses.append(np.correlate(p, matched_filter, mode="same"))
    return filtered_pulses
    

    
