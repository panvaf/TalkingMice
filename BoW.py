"""
Basic bag-of-words models for next token prediction. These are models that do
not consider token ordering in their predictions.
"""

import os
from pathlib import Path
import utils
import numpy as np
import matplotlib.pyplot as plt

# Data directory
data_path = os.path.join(str(Path(os.getcwd()).parent),'data\\03_16_01\\process_audio')
# Filename
bin_size = 100
filename = 'token_seqBin'+str(bin_size)+'.npz'

# Load the data from CSV
data = np.load(os.path.join(data_path,filename))
l_tokens = data['l_tokens']
r_tokens = data['r_tokens']

# First, try to predict what each animal says independently by majority vote
window_szs = np.array([2,5,10,20])
plot = [False,True,False,False]

acc_l = np.zeros(len(window_szs))
acc_r = np.zeros(len(window_szs))

for i, window_size in enumerate(window_szs):

    acc_l[i], pred_l, nonzero_l = utils.majority_prediction(l_tokens, window_size)
    acc_r[i], pred_r, nonzero_r = utils.majority_prediction(r_tokens, window_size)
    
    if plot[i]:
        counts_l, bins = utils.token_hist(l_tokens,'Ground truth')
        counts_l_pr, _ = utils.token_hist(pred_l,'Predictions, window size = {}'.format(window_size),bins)
        counts_l_pr, _ = utils.token_hist(pred_l[nonzero_l],'Predictions (token present), window size = {}'.format(window_size),bins)
        print('Accuracy on left mouse is {} %, window size = {}.'.format(round(100*acc_l[i],2),window_size))
        print('Empty tokens are {} % of total.'.format(round(100*counts_l[0]/sum(counts_l),2)))

        counts_r, bins = utils.token_hist(r_tokens,'Ground truth')
        counts_r_pr, _ = utils.token_hist(pred_r,'Predictions, window size = {}'.format(window_size),bins)
        counts_r_pr, _ = utils.token_hist(pred_r[nonzero_r],'Predictions (token present), window size = {}'.format(window_size),bins)
        print('Accuracy on right mouse is {} %, window size = {}'.format(round(100*acc_r[i],2),window_size))
        print('Empty tokens are {} % of total.'.format(round(100*counts_r[0]/sum(counts_r),2)))
        

# plot accuracies
plt.plot(window_szs,acc_l,label='left')
plt.plot(window_szs,acc_r,label='right')
plt.axhline(1/7,c='red',label='chance')
plt.xscale('log')
plt.xticks(window_szs,window_szs*bin_size/1e3)
plt.xlabel('Context length (s)')
plt.ylabel('Accuracy')
plt.legend()
plt.show()