"""
Get distribution of distances independent of visualizations.
"""

import os
import numpy as np
from scipy.io import loadmat

def load_and_merge_data_from_mat_files(directory_path):
    merged_array = []

    # Get a list of all .mat files in the directory
    mat_files = [file for file in os.listdir(directory_path) if file.endswith('.mat')]

    # Extract data from individual files and merge    
    for i, file_name in enumerate(mat_files):
        
        print("Mat file {} out of {}".format(i+1,len(mat_files)))
        
        file_path = os.path.join(directory_path, file_name)
        
        # Load the .mat file
        data = loadmat(file_path)
        
        # Extract locations
        l_loc = data['l_locations_world']
        r_loc = data['r_locations_world']
        l_x = l_loc[:,4,0]; l_y = l_loc[:,4,1]
        r_x = r_loc[:,4,0]; r_y = r_loc[:,4,1]
        distances = np.sqrt((l_x-r_x)**2 + (l_y-r_y)**2)/1e3
        
        # Merge the array into the main array (you can use np.concatenate or list.extend depending on the array shape)
        merged_array = np.concatenate((merged_array, distances), axis=0) # Adjust the axis parameter if needed

    return merged_array

# Replace 'data_folder' with your actual directory path
directory_path = '//Singingmouse/data/usv_calls/usv_note_analysis/03_div_cage_group01_16-23/location data'
distances = load_and_merge_data_from_mat_files(directory_path)

# Save the merged data to a .npy file
output_file_path = '/distances.npy'
np.save(directory_path+output_file_path, distances)

print("Merged data saved to:", directory_path+output_file_path)