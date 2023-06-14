"""
Try different emdedding methods for USVs.
"""

import os
import glob
from pathlib import Path
import math
import utils
import pandas as pd
import numpy as np
import audio_utils.io
import audio_utils.visualization
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Data directory
data_path = os.path.join(str(Path(os.getcwd()).parent),'data\\03_16_01')
# USV detection filename
filename = 'usv_detections_assigned_230209_repertoire.csv'

# Parameters
win = .1          # Window around USVs being considered, in s

# Read audio files
l_audio_filepath = glob.glob(f'{os.path.join(data_path, "process_audio")}/l_*.wav')[0]
l_avisoft_data, l_avisoft_sr = audio_utils.io.read_avisoft_audio(l_audio_filepath)
r_audio_filepath = glob.glob(f'{os.path.join(data_path, "process_audio")}/r_*.wav')[0]
r_avisoft_data, r_avisoft_sr = audio_utils.io.read_avisoft_audio(r_audio_filepath)
s_audio_filepath = glob.glob(f'{os.path.join(data_path, "process_audio")}/s_*.wav')[0]
s_avisoft_data, s_avisoft_sr = audio_utils.io.read_avisoft_audio(s_audio_filepath)

session_duration = math.ceil(len(l_avisoft_data)/l_avisoft_sr)

# Read USV detection file
usv_detections_filepath = os.path.join(data_path,'process_audio\\',filename)
usv_detections = audio_utils.io.read_usv_detections(usv_detections_filepath,dropna=False)

# mask usv detections by song detections and other filtering
l_usv_detections = usv_detections[(usv_detections['in_song'] == False) & \
                                  (usv_detections['detection_side'] == 'left') & \
                                  (usv_detections['duration'] < int(1e3*win))]

l_usv_detections.dropna(subset=['manual_type'],inplace=True)

'''
n_usv_notes = l_usv_detections.shape[0]
# set some plotting params
n_rows = 10
n_cols = 10

fig, axes = plt.subplots(figsize=(8,8), nrows=n_rows, ncols=n_cols)
for j in range(n_rows):
    for k in range(n_cols):
        i_note = n_rows*n_cols+j*n_cols+k
        if i_note < n_usv_notes:
            t_start = l_usv_detections.iat[i_note, 0]
            t_stop = l_usv_detections.iat[i_note, 1]
            t_align = (t_start+t_stop)/2
            audio_utils.visualization.spectrogram_usv(axes[j,k], l_avisoft_data, l_avisoft_sr, t_align)
        else:
            axes[j,k].set_xticks([])
            axes[j,k].set_yticks([])
            
plt.tight_layout()
plt.savefig('examples.png',bbox_inches='tight',format='png',dpi=300)
plt.show()
'''

# Get data

# Get new index to loop through
l_usv_detections = l_usv_detections.reset_index(drop=True)

# Get labels
labels = l_usv_detections.manual_type

for index, row in l_usv_detections.iterrows():
    
    t_start = row['start']
    t_stop = row['xEnd']
    t_align = (t_start+t_stop)/2

    s_start = int((t_align-win/2)*l_avisoft_sr)
    s_stop = int((t_align+win/2)*l_avisoft_sr)

    data_trial = l_avisoft_data[s_start:s_stop]
    
    Z, freqs, t, extent = audio_utils.visualization.compute_specgram(data_trial, l_avisoft_sr)
    
    if not index:
        specs = np.zeros((len(l_usv_detections),Z.shape[0],Z.shape[1]))
    
    specs[index,:] = Z
    #plt.imshow(Z)
    #plt.show()
    
X = specs.reshape(specs.shape[0],-1)

# Create a PCA instance with the desired number of components
pca = PCA(n_components=100)

# Fit the data to the PCA model and perform dimensionality reduction
X_pca = pca.fit_transform(X)

# Access the principal components
components = pca.components_

# Access the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Print the transformed data and other results
print("Transformed data:")
print(X_pca)
print("Principal components:")
print(components)
print("Explained variance ratio:")
print(explained_variance_ratio)

# Create a figure and a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Generate a range of distinct colors using a colormap
colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow']

# Loop through each label and corresponding coefficients
for label, coefficient in zip(labels, X_pca):
    # Get the index of the label
    label_index = int(label % 12)
    
    # Plot the coefficients with the corresponding color
    ax.scatter(coefficient[0], coefficient[1], coefficient[2], c=colors[label_index], alpha = .2)

# Set labels for the x, y, and z axes
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

# Show the 3D scatter plot
plt.show()