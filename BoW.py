"""
Basic bag-of-words models for next token prediction. These are models that do
not consider token ordering in their predictions.
"""

import os
from pathlib import Path
import utils
import numpy as np

# Data directory
data_path = os.path.join(str(Path(os.getcwd()).parent),'data\\03_16_01\\process_audio')
# Filename
filename = 'token_seq.npz'

# Load the data from CSV
data = np.load(os.path.join(data_path,filename))

# Convert to list
l_tokens = data['l_tokens']
r_tokens = data['r_tokens']

# First, try to predict what each animal says independently by majority vote
window_size = 2
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