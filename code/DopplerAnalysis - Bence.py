# -*- coding: utf-8 -*-
"""
Created on 18-01-2024

@author: Richard, Bence
"""
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from sklearn import preprocessing
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pywt
import statistics
from matplotlib.colors import LinearSegmentedColormap
import math


conv_echo = 1.48 / 160  #Conversion from sample to distance (mm)
prf = 5000              #Pulse Repetition Rate = 5 kHz
f0 = 4e6                #Probe frequency = 4 MHz
fs = 80e6               #Sampling frequency = 80 MHz
c = 1540                #Speed of sound
angle = 45              #Angle of probe in degrees
v_max = c/4 * (prf/f0) * 100/math.cos(angle)    #Max detectable velocity in cm/s

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
    
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        
        header = next(reader)  # Read the header
        if not header:
            return
        
        # Check if header matches expected format
        expected_header = ["XPOS","IDX","TIME","DACVAL","OFFSET"] + [f"D[{i}]" for i in range(7168)]
        if header != expected_header:
            raise ValueError("Unexpected header format")

        xpos = "0"
        for row in reader:
            if row[0] != xpos: 
                positions.append(data)
                data = []
                xpos = row[0]
            data.append([float(val) for val in row[5:]])
        positions.append(data)
                    
            
    return positions   

def reliability_calculator(signal1, signal2):
    sign1 = np.sign(signal1)
    sign2 = np.sign(signal2)
    corr_res = np.correlate(sign1, sign2, mode='same')
    corr_res /= len(sign1)
    rel_index = max(corr_res)
    return rel_index

def time_shift(signal_1, signal_2):
    # Perform cross-correlation for each segment
    number_of_segments = 4
    segment_size = int(512 / number_of_segments)
    shifts_in_region = []
    for i in range(0,number_of_segments):
        segment1 = signal_1[i * segment_size : (i+1) * segment_size]
        segment2 = signal_2[i * segment_size : (i+1) * segment_size]

        # Calculate cross-correlation for the current segment
        cross_corr_results = np.correlate(segment1, segment2, mode='same')
        peaks = np.where(cross_corr_results == np.max(cross_corr_results))[0][0]
        shift = (segment_size / 2 - peaks)
        shifts_in_region.append(shift)
    return max(shifts_in_region, key=abs)

def calculate_velocity(section, plotting=False):
    N = 512 
    pulse_delay = 1
    pulses = []
    for i in range(0,14):
        #Extracting pulses
        bin_i = section[i * N : (i + 1) * N]
        pulses.append(bin_i)
    if plotting:
        figure()
        for i in range(len(pulses)):
            plt.plot(pulses[i], label=f"{i}.pulse")
            plt.title("14 pulses in a section")
            plt.legend()

    
    #Measure the time-shift betwen pulses   
    if plotting: figure()
    shifts = []
    for j in range(len(pulses)-pulse_delay):
        
        i = 0
        signal_1 = pulses[j]
        signal_2 = pulses[j+pulse_delay]
        if reliability_calculator(signal_1, signal_2) > 0.6:
            shifts.append(time_shift(signal_1, signal_2))
        else: shifts.append(0)
            
    # if plotting:
    #     plt.legend()
    #     plt.title("Cross-correlation results")
    #     plt.axvline(x=segment_size / 2, color='r', linestyle='--', label='Zero shift')
    #     figure()
    #     plt.plot(signal_1, label = "1")
    #     plt.plot(signal_2, label = "2")
    #     plt.legend()
    
    #Remove outliers
    med = np.median(shifts)
    filtered_shifts = [i for i in shifts if abs(i - med) <= 10]  
    # filtered_shifts = shifts
    
    if plotting:
        figure()
        plt.plot(filtered_shifts)
        plt.title("Shift between pulses in segment")
        
    if len(filtered_shifts) > 0: shift_avg = sum(filtered_shifts) / len(filtered_shifts)
    else: shift_avg = 0 
        
    #Estimate velocity based on time-shift
    velocity = (c/2) * (shift_avg/fs) * (prf/pulse_delay) * 100 / math.cos(angle)
    if abs(velocity) > v_max: velocity = 0
    return velocity


if __name__ == "__main__":
    
    plt.close('all')
    
    #Open data file
    filename = find_file()
    pos = read_csv(filename)    #Getting a list of lists: 32 segments for 10 positions
    
    only_section = False
    
    if only_section:
        segment = pos[0][7]
        figure()
        plt.plot(segment)
        print(round(calculate_velocity(segment, plotting=True), 2), " cm/s")
    
    else:
        #Create Doppler Flow Image
        figure()
        image = []
        for i in pos:
            doppler = []
            for j in i: 
                shift = calculate_velocity(j)
                doppler.append(shift)
            image.append(doppler)
            
            
        # Define custom colormap
        colors = [(0, 0, 1), (0, 0, 0), (1, 0, 0)]  # Red, Black, Blue
        n_bins = 100  # You can adjust this based on your data distribution
        cmap_name = "doppler_cmap"
        custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
        display = np.array(image).T
        v_max = max(display.max(), abs(display.min()))
        v_max = 20
        plt.imshow(display, cmap=custom_cmap, vmin=-v_max, vmax=v_max)
        plt.title("Doppler Color Flow Image")
        plt.ylabel("Depth segment")
        plt.xlabel("X position (mm)")
        plt.colorbar(label='Velocity (cm/s)')
