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

def find_file():
        
    #Open file dialog to pick the desired BoM
    root = Tk()
    root.attributes('-topmost', True)
    root.iconify() 
    file_path = askopenfilename(title='Select M-mode data', parent=root)
    print("File selected: ", file_path)
    root.destroy()
    return file_path

def read_csv(file_name, start_row, end_row):
    data = []
    
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        
        header = next(reader)  # Read the header
        if not header:
            return
        
        # Check if header matches expected format
        expected_header = ["XPOS","IDX","TIME","DACVAL","OFFSET"] + [f"D[{i}]" for i in range(7168)]
        if header != expected_header:
            raise ValueError("Unexpected header format")

        # Skip rows until the start_row
        for _ in range(start_row - 1):
            next(reader, None)  # None handles cases where we run out of rows
        
        # Read the data from start_row to end_row
        for current_row_number in range(start_row, end_row + 1):
            try:
                row = next(reader)
            except StopIteration:
                break  # end of file reached

            data.append([float(val) for val in row[0:]])
            
    return data

N = 512
filename = find_file()

for index in range(1,32):
    figure(figsize=(40, 4), dpi=600)
    data = read_csv(filename,index,index)
    ll = []
    for e in data:
        debug = (e[2] == 296749);
        f = e[5:]    
        m = statistics.median(f)
        for i in range(0,14):
            l=[]
            #Extracting bins with width = N
            signal = f[i * N : (i + 1) * N]
            plt.plot(signal,linewidth=0.5)
            
            #Finding the zero-crossings
            for j in range(N-1):
                a = signal[j]-m
                b = signal[j+1]-m
                if ( ((a < 0.0) and (b >= 0.0)) or ((a > 0.0) and (b <= 0.0)) ) :
                    l.append(j+1/(1-b/a))   #Linear interpolation
            ll.append(l)
            
    sums = []
    plt.axhline(y=statistics.median(f), color='black', linestyle='-',linewidth=0.5)
        
    for k in range(len(l)):
        plt.axvline(x=l[k], linestyle='-',linewidth=0.5)        
    
    
    plt.title(filename 
              + ", XPOS="+str(data[0][0]) + " [mm]"
              + ", IDX="+str(int(data[0][1])) 
              + ", TIME="+str(int(data[0][2])) + " [us]"
              + ", DACVAL="+str(int(data[0][3])) 
              + ", OFFSET="+str(int(data[0][4])))
    #plt.show()
    plt.savefig(filename 
              + ", XPOS="+str(data[0][0]) + "mm"
              + ", IDX="+str(int(data[0][1])) 
              + ", TIME="+str(int(data[0][2])) + "us"
              + ", DACVAL="+str(int(data[0][3])) 
              + ", OFFSET="+str(int(data[0][4]))+".png")

