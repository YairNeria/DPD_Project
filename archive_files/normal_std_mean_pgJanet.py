import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from scipy.io import loadmat
from pg_janet import PGJanetRNN

# ===================== Dataset =====================
class nMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, y_pred, y_true):
        mse = torch.sum((y_pred - y_true) ** 2)
        denom = torch.sum(y_true ** 2) + 1e-8
        return mse / denom

class PGJanetSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, mat_path='for_DPD.mat', seq_len=10, normalize_amplitude=True):
        mat = loadmat(mat_path, squeeze_me=True)
        X = mat['TX1_BB']
        Y = mat['TX1_SISO']
        
        if normalize_amplitude:
            X_m, X_s = np.mean(X), np.std(X)
            Y_m, Y_s = np.mean(X), np.std(Y)
            X = (X - X_m) / X_s
            Y = (Y - Y_m) / Y_s
        
        amplitudes = np.abs(X).astype(np.float32)
        phases = np.angle(X).astype(np.float32)
        targets = np.stack([Y.real, Y.imag], axis=-1).astype(np.float32)
        
        N = len(amplitudes)
        self.x_abs_seq = []
        self.theta_seq = []
        self.target_seq = []
        for i in range(N - seq_len):
            self.x_abs_seq.append(amplitudes[i:i+seq_len])
            self.theta_seq.append(phases[i:i+seq_len])
            self.target_seq.append(targets[i:i+seq_len])
        self.x_abs_seq = torch.from_numpy(np.array(self.x_abs_seq))
        self.theta_seq = torch.from_numpy(np.array(self.theta_seq))
        self.target_seq = torch.from_numpy(np.array(self.target_seq))

    def __len__(self):
        return self.x_abs_seq.shape[0]
    
    def __getitem__(self, idx):
        return self.x_abs_seq[idx].unsqueeze(-1), self.theta_seq[idx].unsqueeze(-1), self.target_seq[idx]

# ===================== Training Setup =====================
seq_len = 12
hidden_size = 16
n_epochs = 20
batch_size = 64

# ===== With amplitude normalization =====
dataset_norm = PGJanetSequenceDataset('for_DPD.mat', seq_len=seq_len, normalize_amplitude=True)
loader_norm = DataLoader(dataset_norm, batch_size=batch_size, shuffle=True)
model_norm = PGJanetRNN(hidden_size=hidden_size)
loss_function_norm = nMSELoss()
optimizer_norm = optim.Adam(model_norm.parameters(), lr=1e-2)

# --- Scheduler (with normalization) ---  
scheduler_norm = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_norm, mode='min', patience=1, factor=0.5, threshold=1e-4, min_lr=1e-6
)  

# ===== Without amplitude normalization =====
dataset_no_norm = PGJanetSequenceDataset('for_DPD.mat', seq_len=seq_len, normalize_amplitude=False)
loader_no_norm = DataLoader(dataset_no_norm, batch_size=batch_size, shuffle=True)
model_no_norm = PGJanetRNN(hidden_size=hidden_size)
loss_function_no_norm = nMSELoss()
optimizer_no_norm = optim.Adam(model_no_norm.parameters(), lr=1e-2)

# --- Scheduler (no normalization) ---  
scheduler_no_norm = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_no_norm, mode='min', patience=1, factor=0.5, threshold=1e-4, min_lr=1e-6
)

# ===================== Training Loop =====================

# ----- No normalization -----
train_losses_no_norm = []
lrs_no_norm = []
for epoch in range(n_epochs):
    epoch_loss = 0.0
    for batch_x_abs, batch_theta, batch_targets in loader_no_norm:
        outputs = model_no_norm(batch_x_abs, batch_theta)
        loss = loss_function_no_norm(outputs, batch_targets)
        optimizer_no_norm.zero_grad()
        loss.backward()
        optimizer_no_norm.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(loader_no_norm)
    scheduler_no_norm.step(avg_loss)
    train_losses_no_norm.append(avg_loss)
    lrs_no_norm.append(optimizer_no_norm.param_groups[0]['lr'])
    print(f"[NoNorm] Epoch {epoch+1}, Loss: {avg_loss:.6f}, LR: {optimizer_no_norm.param_groups[0]['lr']:.6f}")

# ----- With normalization -----
train_losses_norm = []
lrs_norm = []
for epoch in range(n_epochs):
    epoch_loss = 0.0
    for batch_x_abs, batch_theta, batch_targets in loader_norm:
        outputs = model_norm(batch_x_abs, batch_theta)
        loss = loss_function_norm(outputs, batch_targets)
        optimizer_norm.zero_grad()
        loss.backward()
        optimizer_norm.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(loader_norm)
    scheduler_norm.step(avg_loss) 
    train_losses_norm.append(avg_loss)
    lrs_norm.append(optimizer_norm.param_groups[0]['lr'])
    print(f"[Norm] Epoch {epoch+1}, Loss: {avg_loss:.6f}, LR: {optimizer_norm.param_groups[0]['lr']:.6f}")

# ===================== Plotting =====================
plt.figure(figsize=(8, 5))
plt.plot(10 * np.log10(train_losses_no_norm), label='No Normalization')
plt.plot(10 * np.log10(train_losses_norm), label='With Normalization')
plt.xlabel('Epoch')
plt.ylabel('Training Loss (nMSE) [dB]')
plt.title('Training Loss per Epoch (dB Scale)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
