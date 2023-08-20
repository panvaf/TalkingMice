"""
Convert event data to time series of tokens.
"""

import os
from pathlib import Path
import utils
import pandas as pd
import numpy as np

# Data directory
root = '//Singingmouse/data/usv_calls/usv_note_analysis/03_div_cage_group01_18_song_empty'
data_path = os.path.join(root,'all detections file')
# Filename
filename = 'locations_latents_all_PC_tOrd'

# Parameters
bin_size = 50  # in msec

# Output name
outname = 'token_seq'+'Bin'+str(bin_size)+'Latents'+'.npz'

# Mapping from IDs to tokens, grouping multiple IDs to one token
#tokenizer = {1:1,2:2,3:2,4:2,5:2,6:3,7:3,8:4,9:5,10:6,11:7,12:0}
# Here we treat the "merged" token as if nothing was there. Maybe modify

# Load the data from CSV
data = pd.read_csv(os.path.join(data_path,filename+'.csv'))
#data = data.dropna(subset=['manual_type'])

# Convert start and end values to datetime
data['start'] = pd.to_datetime(data['start'], unit='s')
data['end'] = pd.to_datetime(data['end'], unit='s')

# ID column
#data = data.rename(columns={'manual_type': 'ID'})

# Find the minimum and maximum dates
min_date = data['start'].min().floor(str(bin_size) + 'L') - pd.Timedelta(minutes=1)
max_date = data['end'].max().ceil(str(bin_size) + 'L')

# Create a continuous time series
time_series = pd.date_range(start=min_date, end=max_date, freq=str(bin_size) + 'L')

# Split into left and right
l_data = data[data.Left == 1]
r_data = data[data.Left == 0]

# Empty latent
empty_lat = np.load(os.path.join(data_path,'empty_latent.npz'))['PCs']

# Get time dictionary of tokens
l_bin_tokens, l_ind = utils.serialize_latents(l_data,time_series,empty_lat)
r_bin_tokens, r_ind = utils.serialize_latents(r_data,time_series,empty_lat)

# Convert to array
#l_tokens = np.array(l_bin_tokens); l_ind = np.array(l_ind)
#r_tokens = np.array(r_bin_tokens); r_ind = np.array(r_ind)

# Save arrays of tokens
# np.savez(data_path+'\\'+outname,l_tokens=l_bin_tokens,l_ind=l_ind,r_tokens=r_bin_tokens,r_ind=r_ind)