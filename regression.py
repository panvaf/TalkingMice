"""
Simple regression-based prediction of next token.
"""

import os
from pathlib import Path
import numpy as np
import utils
from torch.utils.data import DataLoader

# Data directory
data_path = os.path.join(str(Path(os.getcwd()).parent),'data\\03_16_01\\process_audio')
# Filename
bin_size = 50
filename = 'token_seqBin'+str(bin_size)+'.npz'

# Load the data from CSV
data = np.load(os.path.join(data_path,filename))
l_tokens = data['l_tokens']
r_tokens = data['r_tokens']

# Predict single animal from itself
window_width = 5
minus_one = False
series = [1, 2, 3, 4, 5, 6, 7]
dataset = utils.OnePredictsOne(series,window_width,minus_one)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Iterate over the data loader
for input_data, target in dataloader:
    print("Input Data:")
    print(input_data)
    print("Target:")
    print(target)
    print()

series1 = [1, 2, 3, 4, 5, 6, 7]
series2 = [7, 8, 9, 10, 11, 12, 13]

dataset = utils.BothPredictBoth(series1,series2,window_width,minus_one)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Iterate over the data loader
for (input_data1, input_data2), (target1, target2) in dataloader:
    print("Input Data 1:")
    print(input_data1)
    print("Target 1:")
    print(target1)
    print("Input Data 2:")
    print(input_data2)
    print("Target 2:")
    print(target2)
    print()