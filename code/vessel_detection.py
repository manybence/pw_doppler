# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 2022

@author: Bence Mány, Neurescue

RAPID project

Detecting a vessel in ultrasound picture
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.util import img_as_float, img_as_ubyte
from skimage.filters import threshold_otsu, median, gaussian, prewitt
from scipy import signal
from scipy.signal import hilbert


intensity_max = 2200
intensity_min = 1900
distance_min = 1000
distance_max = 4000


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def signal_processing(signal):
    fs = 80e6
    filtered = butter_highpass_filter(signal, 5e5, fs)
    
    return filtered

def filter_intenisty(image):
    img_filtered = []
    for i in image:
        line = []
        for j in i:
            if j > intensity_max:
                line.append(intensity_max)
            elif j < intensity_min:
                line.append(intensity_min)
            else:
                line.append(j)
        img_filtered.append(line)
    img_filtered = np.array(img_filtered)
    
    return img_filtered

def find_extremes(image):
    #Compute min and max values of an image
    darkest = 5000
    lightest = 0
    for i in image:
        for j in i:
            if j > lightest: lightest = j
            if j < darkest: darkest = j
            
    print(f'Highest value: {lightest}, lowest value: {darkest}')
    return(darkest, lightest)

def histogram_stretch(image):
    
    img_stretched = []
    max_pixel = int(find_extremes(image)[1])
    min_pixel = int(find_extremes(image)[0])
    for i in image:
        line = []
        for j in i:
            line.append((255 / (max_pixel - min_pixel)) * (j - min_pixel))
        img_stretched.append(line)
    img_stretched = np.array(img_stretched)
    #img_stretched = img_as_ubyte(img_stretched)
    return img_stretched

def gamma_map(image, gamma):
    img_float = img_as_float(image)
    img_scaled = []
    for i in img_float:
        line = []
        for j in i:
            line.append(j ** gamma)
        img_scaled.append(line)
    img_scaled = np.array(img_scaled)
    img_scaled = img_as_ubyte(img_scaled)
    return(img_scaled)
  
def threshold_image(image, threshold):
    
    #Or just: image = image > threshold?
    img_t = []
    for i in image:
        line = []
        for j in i:
            if j > threshold:
                line.append(255)
            else:
                line.append(0)
        img_t.append(line)
    img_t = np.array(img_t)
    img_t = img_as_ubyte(img_t)
    return(img_t)
          
def load_image(path):
    #Load ultrasound scan picture
    us_file = open(path)
    data = np.loadtxt(us_file, delimiter=",")
    pos = data[0]
    frame_org = data[1:]
    return pos, frame_org

def preprocess_image(image, display = False):
    
    #TODO: Add band pass filter + time-gain compensation?
    
    signals = []
    for i in range(image.shape[1]):
        signal_raw = [j[i] for j in image]
    
        #Apply signal processing
        signal_norm = [(x - 2000) for x in signal_raw]
        signal_hilbert = hilbert(signal_norm)
        signal_envelope = np.abs(signal_hilbert)
        signals.append(signal_envelope)

    transposed = np.array(signals).T
    
    if display:
        fig = plt.figure()
        plt.plot(signal_raw)
        plt.title("Single-shot ultrasound signal (raw)")
    
        fig = plt.figure()
        plt.plot(signal_hilbert, label='signal')
        plt.plot(signal_envelope, label='envelope')
        plt.legend()
        plt.title("Single-shot ultrasound signal (Hilbert transformed)")
    
    return transposed

def edge_detection(image, filter_size, filter_type = "med"):
    
    if not filter_size > 0:
        filter_size = 1
        
    if filter_type == "med":
        #Apply median filter
        footprint = np.ones([filter_size, filter_size])
        filtered = median(image, footprint)
        
    if filter_type == "gauss":
        #Apply gaussian filter
        sigma = filter_size
        filtered = gaussian(image, sigma)
    
    #Apply Prewitt edge detection
    edge = prewitt(filtered)
    
    # Calculate Otsu threshold
    threshold = threshold_otsu(edge)
    img_t = threshold_image(edge, threshold)
    
    #Display the detected edges
    min_val = img_t.min()
    max_val = img_t.max()
    
    return img_t

def display(image, title):
    fig = plt.figure()
    plt.imshow(image, extent=(-0.5, image.shape[1]+10000, image.shape[0], -0.5))
    plt.title(title)

def process_image():
    
    #Loading the ultrasound image
    path = "data/scan_22-10-31_155752.csv"
    frame_org = load_image(path)[1]
    display(frame_org, "Original image")

    #Preprocess image (Hilbert transform + envelope)
    frame = preprocess_image(frame_org)
    display(frame, "Processed image")
    
    # high_pass = signal_processing(signal)    
    # fig = plt.figure()
    # plt.plot(high_pass)
    # plt.title("Single-shot ultrasound signal (filtered)")
    
    #Histogram of image
    plt.figure()
    plt.hist(frame.ravel(), bins=256)
    plt.title('Image histogram')


    # #Preprocessing the image
    # frame_abs = []
    # for i in frame:
    #     line = []
    #     for j in i:
    #         line.append(abs(j))
    #     frame_abs.append(line)
        
    # plt.figure()
    # plt.hist(frame_abs.ravel(), bins=256)
    # plt.title('Image histogram')
        
    
    # #Filtering the image by instensity
    # frame = filter_intenisty(frame)
    # frame_filtered = histogram_stretch(frame)
    # display(frame_filtered, "Filtered intensity")
    
    # # #Filter
    # # apply median
    
    # #Otsu thresholding
    # threshold = threshold_otsu(frame_filtered)
    # print(f"Otsu threshold is: {threshold}")
    # frame = frame_filtered > threshold
    # display(frame, "Otsu thresholded")
    
    # #Morphology
    # segmentation.clear_border(frame, 10)
    # footprint = disk(5)
    # closed = closing(frame, footprint)
    # display(closed, "Closing applied")
    # # opened = opening(closed, footprint)
    # # display(opened, "Morphology applied")
    # #outline?
    
    # #BLOB analysis
    
    
    # #Feature filtering
    
    
    # #Edge detection
    # img_t = edge_detection(closed, 25, "med")
    # display(img_t, "Edge detection")
    

    # #Finding the borders
    # fig = plt.figure()
    # object_x = []
    # frame_t = frame.transpose()
    # for i in range(frame.shape[1]):
    #     if max(frame_t[i]) > 0:
    #         object_x.append(i)
            
    # object_y = []
    # for i in range(frame.shape[0]):
    #     if max(frame[i]) > 0:
    #         object_y.append(i)

    # #Find middle position
    # middle = int((min(object_x) + max(object_x)) / 2)
    # print(f"Middle: {middle}")
    
    # #Draw perimeter
    # frame[(min(object_y)-10):min(object_y), min(object_x):max(object_x)] = 250
    # frame[max(object_y):max(object_y)+10, min(object_x):max(object_x)] = 250
    # frame[min(object_y):max(object_y), (min(object_x)-10):min(object_x)] = 250
    # frame[min(object_y):max(object_y), max(object_x):max(object_x)+1] = 250
    # frame[0:3000, middle:(middle+1)] = 250
            
    
    # plt.imshow(frame, extent=(-0.5, frame.shape[1]+10000, frame.shape[0], -0.5))
    # plt.title("Ultrasound scan")
    

    
    


if __name__ == '__main__':
    process_image()
