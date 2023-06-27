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
import matplotlib.pyplot as plt

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
num_epochs = 20

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
    
    
# Plot weight matrices
w = model.linear.weight.data.numpy()
w = w.reshape(output_size,window_width-1,2,n_token)
w_l = w[:,:,0,:]
w_r = w[:,:,1,:]
w_l = np.reshape(w_l,(output_size,n_token*(window_width-1)))
w_r = np.reshape(w_r,(output_size,n_token*(window_width-1)))
w_ll = w_l[:n_token,:]
w_lr = w_l[n_token:,:]
w_rr = w_r[n_token:,:]
w_rl = w_r[:n_token,:]

split_weights = [w_ll, w_lr, w_rl, w_rr]

titles = ['Left-Left', 'Left-Right', 'Right-Left', 'Right-Right']

# Get the maximum and minimum values across all sub-matrices
vmin = np.min([np.min(w) for sublist in split_weights for w in sublist])
vmax = np.max([np.max(w) for sublist in split_weights for w in sublist])

# Plot the sub-matrices one on top of each other
fig, axes = plt.subplots(4, 1, figsize=(8, 5), sharex=True)

for ax, weight, title in zip(axes, split_weights, titles):
    im = ax.imshow(weight, cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
    ax.set_title(title)

# Add x and y labels to the shared axes
axes[-1].set_xlabel('Window size * Input')
axes[len(axes) // 2].set_ylabel('Output')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

# Save plot
plt.savefig('weights.png',bbox_inches='tight',format='png',dpi=300)

# Display the plot
plt.show()