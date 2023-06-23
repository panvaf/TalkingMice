"""
Simple regression-based prediction of next token.
"""

import os
from pathlib import Path
import numpy as np
import utils
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

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
datasetOne = utils.OnePredictsOne(l_tokens,window_width,minus_one)
dataloaderOne = DataLoader(datasetOne, batch_size=16, shuffle=True)

# Predict both animals from both
datasetBoth = utils.BothPredictBoth(l_tokens,r_tokens,window_width,minus_one)
dataloaderBoth = DataLoader(datasetBoth, batch_size=16, shuffle=True)

# Define the model, loss function, and optimizer
n_token = len(set(l_tokens))
input_size = 2 * n_token * (window_width-1)
output_size = 2 * n_token  # Number of unique values in target
model = utils.LinearModel(input_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    for (input_data1, input_data2), (target1, target2) in dataloaderBoth:
        # Concatenate the input data
        input_data = torch.cat((input_data1, input_data2), dim=2)
        
        # Reshape the input data to match the expected input size of the model
        batch_size = input_data.size(0)
        input_data = input_data.view(-1, input_size)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(input_data)
        
        # Correct predictions
        correct1 = torch.argmax(target1, dim=1)
        correct2 = torch.argmax(target2, dim=1)
        
        # Only try to predict non-zeros
        idx1 = correct1 != 0
        idx2 = correct2 != 0
        
        # Compute the loss
        loss1 = criterion(output[idx1, :n_token], correct1[idx1])
        loss2 = criterion(output[idx2, n_token:], correct2[idx2])
        if torch.isnan(loss2):
            loss = loss1
        elif torch.isnan(loss1):
            loss = loss2
        else:
            loss = loss1 + loss2
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        if not torch.isnan(loss):
            running_loss += loss.item()
        
        # Calculate accuracy
        _, predicted1 = torch.max(output[:, :n_token], 1)
        _, predicted2 = torch.max(output[:, n_token:], 1)
        correct_predictions += (torch.sum(predicted1[idx1] == correct1[idx1]) + torch.sum(predicted2[idx2] == correct2[idx2]))
        total_predictions += (torch.sum(idx1) + torch.sum(idx2))
    
    # Print the average loss for each epoch
    avg_loss = running_loss / len(dataloaderBoth)
    accuracy = 100.0 * correct_predictions / total_predictions
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")