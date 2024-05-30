# -*- coding: utf-8 -*-
"""
Created on 18-01-2024

@author: Bence Many
"""

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import doppler_lib as dl


debugging = False

if __name__ == "__main__":
    
    plt.close('all')
    
    #Open data file
    filename = dl.find_file()
    pos = dl.read_csv(filename)    #Getting a list of lists: 32 segments for 10 positions
    
    #Filter signals
    filtered_data = dl.signal_processing(pos)
            
    if debugging:
        #Analyse individual sample
        segment = filtered_data[3][5]
        figure()
        plt.plot(segment)
        print("Detected velocity: ", round(dl.calculate_velocity(segment, plotting=True), 3), " cm/s")
    
    else:
        #Create full Doppler Flow Image
        figure()
        image = []
        for i in filtered_data:
            doppler = []
            for j in i: 
                shift = dl.calculate_velocity(j, plotting=False)
                doppler.append(shift)
            image.append(doppler)
        dl.display_doppler_image(image)
