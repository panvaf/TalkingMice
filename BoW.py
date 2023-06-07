"""
Basic bag-of-words model for next token prediction.
"""
import os
from pathlib import Path
import utils
import pandas as pd
import numpy as np

# Data directory
data_path = os.path.join(str(Path(os.getcwd()).parent),'data\\03_16_01\\process_audio')
# Filename
filename = 'usv_detections_assigned_230209_repertoire'

# Mapping from IDs to tokens, grouping multiple IDs to one token
tokenizer = {1:1,2:2,3:2,4:2,5:2,6:3,7:3,8:4,9:5,10:6,11:7,12:0}

# Load the data from CSV
data = pd.read_csv(os.path.join(data_path,filename+'.csv'))
data = data.dropna(subset=['manual_type'])

# Convert start and end values to datetime
data['start'] = pd.to_datetime(data['start'], unit='s')
data['end'] = pd.to_datetime(data['xEnd'], unit='s')

# ID column
data = data.rename(columns={'manual_type': 'ID'})

# Find the minimum and maximum dates
min_date = data['start'].min().floor('100L')
max_date = data['end'].max().ceil('100L')

# Create a continuous time series
time_series = pd.date_range(start=min_date, end=max_date, freq='100L')

# Split into left and right
l_data = data.loc[(data.detection_side == 'left')]
r_data = data.loc[(data.detection_side == 'right')]

# Get time dictionary of tokens
l_bin_tokens = utils.serialize(l_data, time_series, tokenizer)
r_bin_tokens = utils.serialize(r_data, time_series, tokenizer)
