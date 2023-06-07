"""
Utilities.
"""

import pandas as pd

def serialize(data,time_series,tokenizer):
    '''
    Parameters:
        data: CSV file with event start, end and ID
        time_series: DatetimeIndex file with time bin starts
        tokenizer: ID Category to token dictionary

    Returns
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
            max_duration = pd.Timedelta(0.01)
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