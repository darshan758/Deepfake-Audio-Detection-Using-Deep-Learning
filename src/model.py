import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128, num_layers=2, output_dim=2):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # 2 for bidirectional

    def forward(self, x):
        # x shape: (batch, time, n_mfcc)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # ensure batch dimension
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # take last time step
        out = self.fc(out)
        return out
