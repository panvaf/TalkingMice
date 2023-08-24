"""
Utilities.
"""

import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn


def serialize(data,time_series,tokenizer):
    '''
    Converts raw event-based data to token sequences.
    
    Parameters:
        data: CSV file with event start, end and ID
        time_series: DatetimeIndex file with time bin starts
        tokenizer: ID Category to token dictionary
    
    Returns:
        bin_tokens: dictionary with predominant token in each bin
    '''
    
    # Initialize a dictionary to the predominant token in each bin
    bin_tokens = {bin: 0 for bin in range(len(time_series) - 1)}

    # Iterate over the time series and find tokens that overlap with each bin, then
    # assign dominant token to each bin

    for i in range(len(time_series) - 1):
        # Define bin extent
        bin_start = time_series[i]
        bin_end = time_series[i + 1]
        
        # Find events that fall within that bin
        events_within_bin = data.loc[(data['start'] < bin_end) & (data['end'] > bin_start)]
        
        if not events_within_bin.empty:
            max_duration = pd.Timedelta(0)
            max_duration_event = 0
            
            # Find overlap of events with that specific bin and assign bin to event
            # with biggest overlap
            for _, event in events_within_bin.iterrows():
                event_start = max(bin_start, event['start'])
                event_end = min(bin_end, event['end'])
                duration = event_end - event_start
                
                if duration > max_duration:
                    max_duration = duration
                    max_duration_event = event
            
            bin_tokens[i] = tokenizer[max_duration_event['ID']]
            
    return bin_tokens


def serialize_latents(data, time_series, default_vector):
    '''
    Converts raw event-based data to token sequences.
    
    Parameters:
        data: CSV file with event start, end and latents as final n_lat elements
        time_series: DatetimeIndex file with time bin starts
        default_vector: Predefined latent vector for bins with no events
    
    Returns:
        bin_tokens: Time series with token vectors for each bin
        bin_binary: Binary vector indicating eligibility of bin for prediction
    '''
    
    # Initialize lists to store token vectors and binary indicators
    bin_tokens = []
    bin_binary = []
    
    # Initialize previous latent
    n_lat = len(default_vector)
    prev_latent = np.zeros(n_lat)
    
    # Iterate over the time series and find tokens that overlap with each bin,
    # then assign the corresponding token vector to each bin
    
    for i in range(len(time_series) - 1):
        # Define bin extent
        bin_start = time_series[i]
        bin_end = time_series[i + 1]
        
        # Find events that fall within that bin
        events_within_bin = data.loc[(data['start'] < bin_end) & (data['end'] > bin_start)]
        
        if not events_within_bin.empty:
            max_duration = pd.Timedelta(0)
            max_duration_event = None
            
            # Find overlap of events with that specific bin and assign bin to event
            # with biggest overlap
            for _, event in events_within_bin.iterrows():
                event_start = max(bin_start, event['start'])
                event_end = min(bin_end, event['end'])
                duration = event_end - event_start
                
                if duration > max_duration:
                    max_duration = duration
                    max_duration_event = event
            
            latent = np.array(max_duration_event.iloc[-n_lat:])
            bin_tokens.append(latent)
            
            if np.array_equal(latent,prev_latent):
                # Same event extents to nearby window
                bin_binary.append(0)
            else:
                bin_binary.append(1)
                
            prev_latent = latent
            
        else:
            bin_tokens.append(default_vector)
            bin_binary.append(0)
            
    return np.array(bin_tokens), np.array(bin_binary)



def dataset_split(binary_vector, window_size, splits):
    
    """
    Randomly samples locations in the binary vector where the value is one, looks within a specified
    window size before each sampled location, turns ones in that window into zeros, and continues
    until the target percentage of ones have been turned into zeros.

    Parameters:
        binary_vector (numpy.ndarray): The input binary vector.
        window_size (int): The size of the window to look back before the sampled one.
        splits (float): List with the desired percentage of ones to be turned into zeros.

    Returns:
        list of lists: Each list is the indices for one of the sets, last set being training set.
    """
    
    vec = binary_vector.copy()
    total_ones = np.sum(vec)
    idx_splits = []
    
    for split in splits:
    
        ones_to_remove = int(total_ones * split)    
        result_indices = []
    
        while ones_to_remove > 0:
            one_indices = np.where(vec == 1)[0]
            
            if len(one_indices) == 0:
                break
            
            sampled_index = np.random.choice(one_indices)
            window_start = max(0, sampled_index - window_size + 1)
            window_end = sampled_index + 1
            
            ones_in_window = np.where(vec[window_start:window_end] == 1)[0] + window_start
            result_indices.extend(ones_in_window)
            
            vec[ones_in_window] = 0
            ones_to_remove -= len(ones_in_window)
            
        idx_splits.append(result_indices)
            
    idx_splits.append(np.where(vec == 1)[0])
    
    return idx_splits



def majority_prediction(tokens,window_sz):
    
    '''
    Predicts next token to be the most frequent one in a sliding context window.
    
    Parameters:
        tokens: time series of tokens
        window_sz: sliding window width

    Returns:
        acc: accuracy of next token prediction
        predictions: list of model predictions
        nonzero: mask indicating which predictions have been used to estimate
                 accuracy
    '''
    
    # Create sliding windows of tokens
    windows = [tokens[i:i+window_sz] for i in range(len(tokens) - window_sz + 1)]
    
    predictions = np.empty(len(windows))
    nonzero = np.zeros(len(windows)).astype(bool)
    
    correct_pred = 0
    tokens_present = 0
    
    for i, window in enumerate(windows):
        
        # Count tokens in the window
        token_counts = Counter(window[:-1])
        
        # Find most frequent token
        most_frequent_token = token_counts.most_common(1)[0][0]
        
        # Store predictions to look at statistics
        predictions[i] = most_frequent_token
        
        # Target
        target = window[-1]
        
        # Check if prediction is correct only if a token exists
        if target:
            
            tokens_present += 1
            nonzero[i] = 1
            
            if most_frequent_token == target:
                correct_pred += 1
    
    return correct_pred/tokens_present, predictions, nonzero


def token_hist(tokens,title,bins=None):
    
    '''
    Plot histogram of token occurences.
    
    Parameters:
        tokens: list of numbers representing different tokens
        title: title of histogram
        bins: optional bin argument

    Returns:
        counts: counts of tokens
        bins: bins these counts are based on
    '''
    
    # Create histogram
    if bins is None:
        bins = np.arange(np.min(tokens), np.max(tokens) + 1 + 1e-10, 1)
    
    [counts, bins, _] = plt.hist(tokens, bins=bins)

    # Customize the plot
    plt.xlabel('Token ID')
    plt.ylabel('Count')
    plt.xticks(bins+.5,bins)
    plt.title(title)

    # Display the plot
    plt.show()
    
    return counts, bins


# Create dataset for single animal to predict itself

class OnePredictsOne(Dataset):
    def __init__(self, series, window_width, minus_ones = True):
        self.series = series
        self.window_width = window_width
        self.num_classes = len(set(series))
        self.minus_ones = minus_ones
        
    def __len__(self):
        return len(self.series) - self.window_width
    
    def __getitem__(self, idx):
        window = self.series[idx:idx+self.window_width]
        input_data = window[:-1]
        target = window[-1]
        
        # Apply one-hot encoding to input data
        input_data = torch.tensor(input_data)
        input_data = torch.nn.functional.one_hot(input_data, num_classes=self.num_classes).float()

        # Convert target to one-hot encoding with -1 for non-relevant classes
        target = torch.tensor(target)
        target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).float()
        
        # Convert zeros to -1
        if self.minus_ones:
            input_data[input_data == 0] = -1
            target[target == 0] = -1
        
        return input_data, target
    
    
# Create dataset for both animals to predict both

class BothPredictBoth(Dataset):
    def __init__(self, series1, series2, window_width, minus_ones = True):
        self.series1 = series1
        self.series2 = series2
        self.window_width = window_width
        self.num_classes = len(set(np.concatenate((series1,series2))))
        self.minus_ones = minus_ones
        
    def __len__(self):
        return len(self.series1) - self.window_width
    
    def __getitem__(self, idx):
        window1 = self.series1[idx:idx+self.window_width]
        window2 = self.series2[idx:idx+self.window_width]
        
        input_data1 = window1[:-1]
        input_data2 = window2[:-1]
        
        target1 = window1[-1]
        target2 = window2[-1]
        
        # Apply one-hot encoding to input data
        input_data1 = torch.tensor(input_data1)
        input_data1 = torch.nn.functional.one_hot(input_data1.to(torch.int64), num_classes=self.num_classes).float()
        
        input_data2 = torch.tensor(input_data2)
        input_data2 = torch.nn.functional.one_hot(input_data2.to(torch.int64), num_classes=self.num_classes).float()
        
        # Convert targets to one-hot encoding
        target1 = torch.tensor(target1)
        target1 = torch.nn.functional.one_hot(target1.to(torch.int64), num_classes=self.num_classes).float()
        
        target2 = torch.tensor(target2)
        target2 = torch.nn.functional.one_hot(target2.to(torch.int64), num_classes=self.num_classes).float()
        
        # Convert zeros to -1
        if self.minus_ones:
            input_data1[input_data1 == 0] = -1
            input_data2[input_data2 == 0] = -1
            target1[target1 == 0] = -1
            target2[target2 == 0] = -1
        
        return (input_data1, input_data2), (target1, target2)
    
# Create dataset for both animals to predict one. Cleaner implementation

class BothPredictOne(Dataset):
    def __init__(self, series1, series2, window_width, minus_ones = False,
                 predict = 'First'):
        self.series1 = series1
        self.series2 = series2
        self.window_width = window_width
        self.num_classes = len(set(np.concatenate((series1,series2))))
        self.minus_ones = minus_ones
        self.predict = predict
        
    def __len__(self):
        return len(self.series1) - self.window_width
    
    def __getitem__(self, idx):
        window1 = self.series1[idx:idx+self.window_width]
        window2 = self.series2[idx:idx+self.window_width]
        
        input_data1 = window1[:-1]
        input_data2 = window2[:-1]
        
        if self.predict == 'First':    
            target = window1[-1]
        elif self.predict == 'Second':
            target = window2[-1]
        
        # Apply one-hot encoding to input data
        input_data1 = torch.tensor(input_data1)
        input_data1 = torch.nn.functional.one_hot(input_data1.to(torch.int64), num_classes=self.num_classes).float()
        
        input_data2 = torch.tensor(input_data2)
        input_data2 = torch.nn.functional.one_hot(input_data2.to(torch.int64), num_classes=self.num_classes).float()
        
        # Convert target to one-hot encoding
        target = torch.tensor(target)
        target = torch.nn.functional.one_hot(target.to(torch.int64), num_classes=self.num_classes).float()

        # Convert zeros to -1
        if self.minus_ones:
            input_data1[input_data1 == 0] = -1
            input_data2[input_data2 == 0] = -1
            target[target == 0] = -1
        
        return (input_data1, input_data2), target
    
    
# Create latents dataset for single animal to predict itself

class OnePredictsOneLatent(Dataset):
    def __init__(self, series, idxs, window_width):
        self.series = series
        self.window_width = window_width
        self.idxs = idxs
        
    def __len__(self):
        return np.sum(self.idxs)
    
    def __getitem__(self, idx):
        
        if not self.idxs[idx]:
            print("Sampled in wrong location!")
            return
        
        window = self.series[idx-self.window_width+1:idx+1,:]
        input_data = window[:-1,:].astype(np.float32)
        target = window[-1,:].astype(np.float32)
        
        # Convert to tensors
        input_data = torch.tensor(input_data)
        target = torch.tensor(target)
        
        return input_data, target


# Linear model with softmax output

class LinearModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        out = self.linear(x)
        return nn.functional.softmax(out, dim=1)
    
    
# Append location information to dataframe with USV detections

def append_loc(usv,l_locations,r_locations,fps=50):
    
    # Timestamps
    t_start = usv['start']
    t_end = usv['end']
    
    frame_start = int(t_start*fps)
    frame_end = int(t_end*fps) + 1
    
    # Average locations
    l_loc = np.average(l_locations[frame_start:frame_end,:],axis=0).flatten()
    r_loc = np.average(r_locations[frame_start:frame_end,:],axis=0).flatten()
    
    # Column names
    sensor = ['1', '2', '3', '4', '5', '6']
    coord = ['x', 'y']
    
    names = np.array([[sensor[i] + '_' + coord[j] for j in range(len(coord))] for i in range(len(sensor))]).flatten()
    
    names_l = ['l_' + element for element in names]
    names_r = ['r_' + element for element in names]
    
    # Append to series
    for name, location in zip(names_l, l_loc):
        usv[name] = location
    for name, location in zip(names_r, r_loc):
        usv[name] = location
    
    return usv