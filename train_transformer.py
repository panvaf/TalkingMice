"""
Train transformers. 
"""

import os
import numpy as np
import utils
from torch.utils.data import DataLoader, Subset

# Parameters
bin_size = 50 # in msec
window_size = 1 # in sec
context_win = 60 # in sec
n_tokens = int(window_size/bin_size*1e3)
n_context = int(context_win/bin_size*1e3)
splits = [.15, .15]

# Load data
data_dir = '//Singingmouse/data/usv_calls/usv_note_analysis/03_div_cage_group01_18_song_empty/all detections file'
filename = 'token_seqBin50Latents.npz'
data = np.load(os.path.join(data_dir,filename),allow_pickle=True)
tokens = data['l_tokens']
idxs = data['l_ind']

# Dataset
datasetOne = utils.OnePredictsOneLatent(tokens,idxs,n_tokens)

# Split in train, test and validation sets
np.random.seed(42)
idx_splits = utils.dataset_split(idxs,n_context,splits)

train_dataset = Subset(datasetOne,idx_splits[-1])
val_dataset = Subset(datasetOne,idx_splits[0])
test_dataset = Subset(datasetOne,idx_splits[1])