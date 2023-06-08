"""
Utilities.
"""

import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


def serialize(data,time_series,tokenizer):
    '''
    Converts raw event-based data to token sequences.
    
    Parameters:
        data: CSV file with event start, end and ID
        time_series: DatetimeIndex file with time bin starts
        tokenizer: ID Category to token dictionary

    Returns:
        bin_tokens: dictionary with dominant token in each bin
    '''
    
    # Initialize a dictionary to the dominant token in each bin
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