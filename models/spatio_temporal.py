import sys
sys.path.append("../")

import torch.nn as nn
from einops import rearrange

from models.modules import ISAB, SAB, PMA
from models.networks import build_mlp

class SpatioTemporalLSTM(nn.Module):
    def __init__(self, dim_input=2, num_outputs=1, dim_output=128, dim_hidden=128, num_heads=4, ln=True):
        super(SpatioTemporalLSTM, self).__init__()

        # # Bi-dirctional LSTM encoder
        self.phi = nn.LSTM(input_size=dim_output, hidden_size=dim_hidden, num_layers=8, batch_first=True, bidirectional=True)

        self.enc = nn.Sequential(
                SAB(dim_input, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln))

        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))

    def forward(self, x):

        batch_size = x.shape[0]

        # Combine batch and number of stars into a single 'batch' dimension
        # b = batch, s = star, t = time, xy = angular (x,y) coordinate
        x = rearrange(x, 'b s t xy -> (b t) s xy')
        # !!! Is it wrong to make interleave b and t?

        x = self.enc(x)
        x = self.dec(x).squeeze()  # s dimension eliminated
        # At this point, you have summaries of time sieres 1 at time 1 ...
        # time series N at time L

        x = rearrange(x, '(b t) h -> b t h', b=batch_size)

        x, (h, c) = self.phi(x)  # apply LSTM: t dimension eliminated
        # At this point you have summaries for time series 1 ... time series N

        # Extract two 'hidden' vectors corresponding to two directions in LSTM
        # d = direction, l = layer, b = batch, h = hidden
        # Keep only last layer, but both directions
        h = rearrange(h, '(d l) b h -> b d l h', d=2)[:, :, -1, :]
        h = rearrange(h, 'b d h -> b (d h)')

        # Your summary for each time series is (direc) x (dim-h) dimensional = 2 x dim_hidden

        return h
