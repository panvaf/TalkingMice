"""
Train transformers. 
"""

import os
import numpy as np
import utils
from torch.utils.data import DataLoader, Subset
from transformers import TransformerEncoderOne

# Parameters
bin_size = 100 # in msec
window_size = 5 # in sec
context_win = 60 # in sec
n_tokens = int(window_size/bin_size*1e3)
n_context = int(context_win/bin_size*1e3)
splits = [.15, .15]

# Load data
data_dir = '//Singingmouse/data/usv_calls/usv_note_analysis/03_div_cage_group01_18_song_empty/all detections file'
filename = 'token_seqBin'+str(bin_size)+'Latents.npz'
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

# Dataloader
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

# Create an iterator from the DataLoader
dataloader_iterator = iter(train_dataloader)

# Fetch the next batch of data samples (just one batch)
batch = next(dataloader_iterator)

# Unpack the batch into inputs and targets
inputs, targets = batch

# Define model
transformer = TransformerEncoderOne(d_model=7, n_heads=1, d_ff=32, max_seq_len = 49,
                                    n_layers=3, dropout=0.1, n_hidden=32)

output = transformer(inputs.permute(1,0,2))