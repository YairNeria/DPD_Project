import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from torch.utils.data import DataLoader
from scipy.io import loadmat
from pg_janet import PGJanetRNN

class nMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, y_pred, y_true):
        mse = torch.sum((y_pred - y_true) ** 2)
        denom = torch.sum(y_true ** 2) + 1e-8
        return mse / denom
    
class PGJanetSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, mat_path='for_DPD.mat', seq_len=10):
        mat = loadmat(mat_path, squeeze_me=True)
        X = mat['TX1_BB']
        Y = mat['TX1_SISO']
        amplitudes = np.abs(X).astype(np.float32)
        amplitudes /= np.max(amplitudes)  # <-- Normalize to [0, 1]
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

# Hyperparameters
seq_len = 12
hidden_size = 16
n_epochs = 20
batch_size = 64

# Dataset, DataLoader, Model
dataset = PGJanetSequenceDataset('for_DPD.mat', seq_len=seq_len)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = PGJanetRNN(hidden_size=hidden_size)
loss_function = nMSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)

#former train loop
# # Training loop
# for epoch in range(n_epochs):
#     for batch_x_abs, batch_theta, batch_targets in loader:
#         outputs = model(batch_x_abs, batch_theta)
#         loss = loss_function(outputs, batch_targets)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

# Learning Rate Scheduler (adaptive LR)
# This scheduler reduces the learning rate when a metric has stopped improving.
#mode=minimum means we want to minimize the loss
scheduler = optim.lr_scheduler.ReduceLROnPlateau(  # <-- Adaptive lr
    optimizer, mode='min', patience=2, factor=0.5, threshold=1e-4, min_lr=1e-6)  # <-- Adaptive lr
# Plot training loss and learning rate per epoch

# Track learning rate and loss per epoch
# (Move these lists to before the training loop)
learning_rates = []
epoch_losses_list = []

# Training loop
for epoch in range(n_epochs):
    epoch_loss = 0.0  # <-- Adaptive lr
    for batch_x_abs, batch_theta, batch_targets in loader:
        outputs = model(batch_x_abs, batch_theta)
        loss = loss_function(outputs, batch_targets)
        optimizer.zero_grad()
        # Initialize epoch_losses at the start of each epoch

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()  # <-- Adaptive lr

    epoch_loss /= len(loader)  # <-- Adaptive lr
    scheduler.step(epoch_loss)
    # Then, inside the training loop, after scheduler.step(epoch_loss):
    learning_rates.append(optimizer.param_groups[0]['lr'])
    epoch_losses_list.append(epoch_loss)  # <-- Adaptive lr

    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6e}")

# Save the model
torch.save(model.state_dict(), 'pg_janet_rnn.pth')
import matplotlib.pyplot as plt

# Plot training loss and learning rate per epoch

# Convert NMSE loss to dB for plotting
epoch_losses_db = [10 * np.log10(l) for l in epoch_losses_list]
# After training, plot:
# Plot training loss per epoch
plt.figure()
plt.plot(range(1, n_epochs + 1), epoch_losses_db, label='Training Loss (dB)')
plt.xlabel('Epoch')
plt.ylabel('Loss (dB)')
plt.title('Training Loss per Epoch (dB)')
plt.grid(True)
plt.legend()
plt.show()

# Plot learning rate per epoch
plt.annotate('Normalized input', xy=(0.7, 0.95), xycoords='axes fraction',
             fontsize=10, color='green', ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))

df = pd.DataFrame({'epoch': range(1, n_epochs + 1), 'loss_db': epoch_losses_db, 'learning_rate': learning_rates})
df.to_csv('loss_epoch.csv', index=False)
plt.figure()
plt.plot(range(1, n_epochs + 1), learning_rates, label='Learning Rate')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate per Epoch')
plt.grid(True)
plt.legend()
plt.show()