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
from sklearn.decomposition import PCA, FastICA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

# Data directory
data_path = os.path.join(str(Path(os.getcwd()).parent),'data\\03_16_01')
# USV detection filename
filename = 'usv_detections_assigned_230209_repertoire.csv'

# Mapping from IDs to tokens, grouping multiple IDs to one token
tokenizer = {1:1,2:2,3:2,4:2,5:2,6:3,7:3,8:4,9:5,10:6,11:7,12:0}

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
n_rows = 5
n_cols = 5

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
#plt.savefig('examples.png',bbox_inches='tight',format='png',dpi=300)
plt.show()
'''

# Get data

# Get new index to loop through
l_usv_detections = l_usv_detections.reset_index(drop=True)

# Get labels
labels = [tokenizer[label] for label in l_usv_detections.manual_type]

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

# PCA

# Create a PCA instance with the desired number of components
pca = PCA(n_components=100)

# Fit the data to the PCA model and perform dimensionality reduction
X_pca = pca.fit_transform(X)

# Access the principal components
components_pca = pca.components_

# Access the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Scree plot
plt.plot(np.arange(1,101),explained_variance_ratio)
plt.xlabel('Components')
plt.ylabel('Explained variance %')
plt.title('PCA scree plot')
plt.axvline(3,color='red')
plt.show()

# Create a figure and a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Get the total number of data points
total_points = len(labels)

# Generate a range of distinct colors using a colormap
colors = ['olive', 'red', 'green', 'blue', 'purple', 'cyan', 'magenta', 'yellow']

# Loop through each label and corresponding coefficients
for i in range(0, total_points, 10):
    label = int(labels[i])
    coefficient = X_pca[i,:]
    
    # Plot the coefficients with the corresponding color
    ax.scatter(coefficient[0], coefficient[1], coefficient[2], c=colors[label], alpha = .4, s = 25)
    
# Create a list of labels for the legend
legend_labels = ['Merge','Simple downsweep','Complex downsweep','Trill','Slashy','Song note','Ultra short','Bark']

# Create custom legend handles with specified colors
legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                         label=label, markersize=8) for label, color in zip(legend_labels, colors)]

# Add the legend to the plot
ax.legend(handles=legend_handles, fontsize=14)

# Set labels for the x, y, and z axes
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

# Show the 3D scatter plot
plt.show()


# ICA

# Create an instance of the FastICA model
ica = FastICA(n_components=3, random_state=42)

# Fit the model to the data and extract the independent components
X_ica = ica.fit_transform(X)

# Create a figure and a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Loop through each label and corresponding coefficients
for i in range(0, total_points, 10):
    label = int(labels[i])
    coefficient = X_ica[i,:]
    
    # Plot the coefficients with the corresponding color
    ax.scatter(coefficient[0], coefficient[1], coefficient[2], c=colors[label], alpha = .4, s = 25)

# Add the legend to the plot
ax.legend(handles=legend_handles, fontsize=14)

# Set labels for the x, y, and z axes
ax.set_xlabel('IC1')
ax.set_ylabel('IC2')
ax.set_zlabel('IC3')

# Show the 3D scatter plot
plt.show()