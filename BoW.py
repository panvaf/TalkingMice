"""
Basic bag-of-words models for next token prediction. These are models that do
not consider token ordering in their predictions.
"""

import os
from pathlib import Path
import utils
import pandas as pd

# Data directory
data_path = os.path.join(str(Path(os.getcwd()).parent),'data\\03_16_01\\process_audio')
# Filename
filename = 'usv_detections_assigned_230209_repertoire'

# Mapping from IDs to tokens, grouping multiple IDs to one token
tokenizer = {1:1,2:2,3:2,4:2,5:2,6:3,7:3,8:4,9:5,10:6,11:7,12:0}
# Here we treat the "merged" token as if nothing was there. Maybe modify

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

# Convert to list
l_tokens = list(l_bin_tokens.values())
r_tokens = list(r_bin_tokens.values())

# First, try to predict what each animal says independently by majority vote
window_size = 10
acc_l, pred_l = utils.majority_prediction(l_tokens, window_size)
counts_l, bins = utils.token_hist(l_tokens,'Ground truth')
counts_l_pr, _ = utils.token_hist(pred_l,'Predictions',bins)
print('Accuracy on left mouse is {} %.'.format(round(100*acc_l,2)))
print('Empty tokens are {} % of total.'.format(round(100*counts_l[0]/sum(counts_l),2)))

acc_r, pred_r = utils.majority_prediction(r_tokens, window_size)
counts_r, bins = utils.token_hist(r_tokens,'Ground truth')
counts_r_pr, _ = utils.token_hist(pred_r,'Predictions',bins)
print('Accuracy on right mouse is {} %'.format(round(100*acc_r,2)))
print('Empty tokens are {} % of total.'.format(round(100*counts_r[0]/sum(counts_r),2)))