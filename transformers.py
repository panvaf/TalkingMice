"""
Main transformer classes.
"""

import torch
import torch.nn as nn

class TransformerEncoderOne(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, max_seq_len, n_layers, dropout, n_hidden):
        super().__init__()
        # Sin-cosine positional embedding for simplicity
        self.positional_encoding = SinCosPosEncoder(d_model, dropout, max_seq_len).to(torch.float)
        # Define encoder layers
        encoder_layers = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers,n_layers)
        # MLP head
        self.MLP_head = nn.Sequential(
            nn.Linear(max_seq_len, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1)
        )
        
    def forward(self, x):
        
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x.permute(2,1,0)
        x = self.MLP_head(x)  # Adjust the output transformation
        x = x.permute(2,1,0)
        
        return x


# Sin-cosine positional encoder

class SinCosPosEncoder(nn.Module):
    def __init__(self, d_model, dropout, max_seq_len):
        super().__init__()
        
        # [part 2a]
        # define ``positional_encoding`` as described in the GoogleDoc instruction and register it 
        
        pe = torch.zeros(max_seq_len, 1, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model+1, 2).float() * (-torch.log(torch.tensor(10000.0)) / (d_model+1)))
        
        pe[:,0,0::2] = torch.sin(position * div_term)
        pe[:,0,1::2] = torch.cos(position * div_term)[:,:-1]
        
        self.register_buffer('positional_encoding', pe)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        Returns:
            output Tensor of shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.positional_encoding[:x.size(0)]
        return self.dropout(x)
