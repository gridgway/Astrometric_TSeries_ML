import sys
sys.path.append("../")

import torch.nn as nn
from einops import rearrange

from models.modules import ISAB, SAB, PMA
from models.networks import build_mlp

class SetTransformerLSTM(nn.Module):
    def __init__(self, dim_input=128, num_outputs=1, dim_output=1, dim_hidden=128, num_heads=4, ln=True):
        super(SetTransformerLSTM, self).__init__()

        # Bi-dirctional LSTM encoder
        self.phi = nn.LSTM(input_size=2, hidden_size=int(dim_hidden / 2), num_layers=4, batch_first=True, bidirectional=True)

        self.enc = nn.Sequential(
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln))

        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))

    def forward(self, x):

        batch_size = x.shape[0]
        
        # Combine batch and number of stars into a single 'batch' dimension
        x = rearrange(x, 'b s t xy -> (b s) t xy')
        
        # Encode all through LSTM
        x, (h, c) = self.phi(x) 
        
        # Extract two 'hidden' vectors corresponding to two directions in LSTM
        h = rearrange(h, '(d l) b h -> b d l h', d=2)[:, :, -1, :]
        h = rearrange(h, 'b d h -> b (d h)')
        
        # Uncombine batch and star dimensions
        x = rearrange(h, '(b s) h -> b s h', b=batch_size)

        return self.dec(self.enc(x))[:, 0, :]

class DeepSetLSTM(nn.Module):
    def __init__(self, hidden_dim=64, output_dim=1):
        """ A deep set model based on an LSTM encoder
        """
        super(DeepSetLSTM, self).__init__()
                    
        # Bi-dirctional LSTM encoder
        self.phi = nn.LSTM(input_size=2, hidden_size=hidden_dim, num_layers=8, batch_first=True, bidirectional=True)
        self.phi_mlp = build_mlp(input_dim=2 * hidden_dim, hidden_dim=4 * hidden_dim, output_dim=2 * hidden_dim, layers=4)
        
        # MLP post-aggregation network
        self.rho = build_mlp(input_dim=2 * hidden_dim, hidden_dim=2 * hidden_dim, output_dim=output_dim, layers=4)

    def forward(self, x):
        
        batch_size = x.shape[0]
        
        # Combine batch and number of stars into a single 'batch' dimension
        x = rearrange(x, 'b s t xy -> (b s) t xy')
        
        # Encode all through LSTM
        x, (h, c) = self.phi(x) 
        
        # Extract two 'hidden' vectors corresponding to two directions in LSTM
        h = rearrange(h, '(d l) b h -> b d l h', d=2)[:, :, -1, :]
        h = rearrange(h, 'b d h -> b (d h)')
        
        h = self.phi_mlp(h)

        # Uncombine batch and star dimensions, then do mean-aggregation along star axis
        # (Enforce permutation invariance)
        h = rearrange(h, '(b s) h -> b s h', b=batch_size).mean(-2)
        
        # Pass through post-aggregation network
        x = self.rho(h)
        
        return x