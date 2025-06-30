import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.io import loadmat

class PGJanetCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PGJanetCell, self).__init__()
        self.hidden_size = hidden_size
        
        # |x_n|, cos(theta_n), sin(theta_n) are each scalar per sample
        # For gate inputs: concatenate [h_{n-1}, input]
        self.amplitude_gate = nn.Linear(hidden_size + 1, hidden_size)
        self.phase1_gate = nn.Linear(hidden_size + 1, hidden_size)
        self.phase2_gate = nn.Linear(hidden_size + 1, hidden_size)
        
        # JANET part
        self.forget_gate = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.candidate = nn.Linear(hidden_size + hidden_size, hidden_size)
        
        # Output layer
        output_size = 2  # For I/Q
        self.output_layer = nn.Linear(hidden_size, output_size)
        # Note: output_size can be adjusted if you want to output I/Q separately

    def forward(self, h_prev, x_abs, cos_theta, sin_theta):
        # Concatenate hidden state with input feature for each gate
        amplitude_input = torch.cat([h_prev, x_abs], dim=1)     # [batch, hidden + 1]
        phase1_input = torch.cat([h_prev, cos_theta], dim=1)    # [batch, hidden + 1]
        phase2_input = torch.cat([h_prev, sin_theta], dim=1)    # [batch, hidden + 1]

        a_n = torch.tanh(self.amplitude_gate(amplitude_input))
        p1_n = torch.tanh(self.phase1_gate(phase1_input))
        p2_n = torch.tanh(self.phase2_gate(phase2_input))

        # Compose non-linear "u_n" as described
        u_n = a_n * p1_n * p2_n * (1 - a_n) * (1 - p1_n) * (1 - p2_n)
        
        # Concatenate h_prev and u_n for the main gates
        concat_hu = torch.cat([h_prev, u_n], dim=1)

        f_n = torch.sigmoid(self.forget_gate(concat_hu))
        g_n = torch.tanh(self.candidate(concat_hu))

        h_n = f_n * h_prev + (1 - f_n) * g_n

        # Output I/Q (real+imag), or could be 2 units if I/Q split
        y_n = self.output_layer(h_n)
        return h_n, y_n

class PGJanetRNN(nn.Module):
    def __init__(self, hidden_size):
        super(PGJanetRNN, self).__init__()
        self.hidden_size = hidden_size
        # input_size=2 for [I,Q] output, or adjust as needed
        self.cell = PGJanetCell(input_size=2, hidden_size=hidden_size)

    def forward(self, x_abs, theta):
        '''
        x_abs: [batch, seq_len] or [batch, seq_len, 1]
        theta: [batch, seq_len] or [batch, seq_len, 1]
        '''
        # If input is 2D, add feature dimension
        if x_abs.dim() == 2:
            x_abs = x_abs.unsqueeze(-1)
        if theta.dim() == 2:
            theta = theta.unsqueeze(-1)
        batch_size, seq_len, _ = x_abs.shape
        h = torch.zeros(batch_size, self.hidden_size, device=x_abs.device)
        outputs = []

        for t in range(seq_len):
            x_t = x_abs[:, t, :]  # [batch, 1]
            theta_t = theta[:, t, :]  # [batch, 1]
            cos_theta = torch.cos(theta_t)
            sin_theta = torch.sin(theta_t)
            
            h, y = self.cell(h, x_t, cos_theta, sin_theta)
            outputs.append(y.unsqueeze(1))

        # Stack outputs to [batch, seq_len, 2]
        return torch.cat(outputs, dim=1)


class PGJanetDataset(Dataset):
    def __init__(self, mat_path='for_DPD.mat'):
        mat = loadmat(mat_path, squeeze_me=True)
        X = mat['TX1_BB']      # Complex baseband input
        Y = mat['TX1_SISO']    # Distorted output

        self.amplitudes = np.abs(X).astype(np.float32)     # [N]
        self.phases = np.angle(X).astype(np.float32)       # [N]
        self.targets = np.stack([Y.real, Y.imag], axis=-1).astype(np.float32)  # [N, 2]

        self.amplitudes = torch.from_numpy(self.amplitudes).unsqueeze(-1)  # [N, 1]
        self.phases = torch.from_numpy(self.phases).unsqueeze(-1)          # [N, 1]
        self.targets = torch.from_numpy(self.targets)                      # [N, 2]
        self.n_samples = self.amplitudes.shape[0]

    def __getitem__(self, idx):
        # Returns: amplitude [1], phase [1], target [2]
        return self.amplitudes[idx], self.phases[idx], self.targets[idx]

    def __len__(self):
        return self.n_samples