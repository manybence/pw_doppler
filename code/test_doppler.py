# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 11:29:00 2024

@author: Bence Many

Test cases for Time-shift velocity estimation
"""
import numpy as np
import matplotlib.pyplot as plt
import doppler_lib as dl
import math
import numpy

pulse_length = 128

def create_signals2(time_shift, plotting=False):
    
    # Parameters
    amplitude = 1
    frequency = 4e6  # 4 MHz
    sample_rate = 80e6  # 80 MHz
    num_samples = int(pulse_length/2)
    
    # Time array and time shifted time aray
    t = np.linspace(0, num_samples / sample_rate, num_samples)
    t_shifted = np.linspace(-time_shift, (num_samples / sample_rate) - time_shift, num_samples)
    
    # Create sinusoidal arrays
    sinusoid_1a = amplitude * np.sin(2 * np.pi * frequency * t)
    sinusoid_1b = amplitude * np.sin(2 * np.pi * frequency * t_shifted)
    sinusoid_2 = numpy.concatenate((sinusoid_1a, sinusoid_1b))
    sinusoid_1 = numpy.concatenate((sinusoid_1a, sinusoid_1a))

    if plotting:
        plt.figure()
        plt.plot(sinusoid_1, label = "shifted")
        plt.plot(sinusoid_2, label = "original")
        plt.xlabel("Time (us)")
        plt.ylabel("Amplitude")
        plt.legend()
    
    return sinusoid_1, sinusoid_2

def create_signals(time_shift, plotting=False):
    
    # Parameters
    amplitude = 1
    frequency = 4e6  # 4 MHz
    sample_rate = 80e6  # 80 MHz
    num_samples = pulse_length
    
    # Time array and time shifted time aray
    t = np.linspace(0, num_samples / sample_rate, num_samples)
    t_shifted = np.linspace(-time_shift, (num_samples / sample_rate) - time_shift, num_samples)
    
    # Create sinusoidal arrays
    sinusoid_1 = amplitude * np.sin(2 * np.pi * frequency * t)
    sinusoid_2 = amplitude * np.sin(2 * np.pi * frequency * t_shifted)

    if plotting:
        plt.figure()
        plt.plot(t, sinusoid_1, label = "original")
        plt.plot(t, sinusoid_2, label = "shifted")
        plt.xlabel("Time (us)")
        plt.ylabel("Amplitude")
        plt.legend()
    
    return sinusoid_1, sinusoid_2
    
def test_time_shift(t_shift):
    
    #Create time-shifted signals, measure shift  
    sinusoid_1, sinusoid_2 = create_signals2(t_shift, False)
    shift = dl.time_shift(sinusoid_1, sinusoid_2)
    dt = round((shift / 80e6) * 1e6, 5)
    error = round((t_shift * 1e6 - dt) / (t_shift * 1e6), 2)
    print(f"\nExpected Shift = {round(t_shift*1e6, 5)} us")
    print(f"Shift = {dt} us")
    print(f"Error = {error * 100} %")
    
    return dt, error
    
def measure_shift_error():
    c = 1540
    prf = 5000
    fs = 80e6
    area = 0.00785 #dm2
    max_flow = 3.2 #l/min
    max_v = max_flow / (60*10*area) #m/s
    angle = 45
    print("Max v = ", max_v)
    max_shift = int(max_v*fs*2 / (c*prf))
    print(f"Max shift = {max_shift} samples")
    
    errors = []
    shifts = np.linspace(-20/80e6, 20/80e6, 200)
    
    for i in shifts:
        dt, error = test_time_shift(i)
        errors.append(abs(error) * 100)
        
    FR = 300*area*c*shifts*prf/math.cos(math.radians(angle))
        
    plt.figure()
    plt.plot(FR, errors)
    plt.title("Error vs Flow rate")
    plt.ylabel("Error (%)")
    plt.xlabel("Flow Rate (liter/min)")
    
    
    
# create_signals2(-5/80e6, True)
measure_shift_error()
# test_time_shift(0.15e-6)