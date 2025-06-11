import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from scipy.io import loadmat
import matplotlib.pyplot as plt
from train_pg_janet_class import PGJanetSequenceDataset, TrainModel
from pg_janet import PGJanetRNN

"""
This script implements the PG-JANET training and integration with Iterative Learning Control (ILC) and DLA fine-tuning,
as described in Fig. 5 of the referenced article (Digital Predistortion of RF Power Amplifiers With Phase-Gated Recurrent Neural Networks).

Stages:
1. Pre-train PG-JANET model on the dataset (as before).
2. ILC Integration: Iteratively update the input to the PA using the trained PG-JANET model and measured PA output.
3. DLA Fine-tuning: Further train the PG-JANET model using the ILC-refined data.
"""

# -----------------------------
# Hyperparameters
# -----------------------------
seq_len = 12
hidden_size = 20
n_epochs = 30
batch_size = 64
val_ratio = 0.2
ilc_iterations = 5  # Number of ILC iterations
fine_tune_epochs = 10  # DLA fine-tuning epochs

# -----------------------------
# Load dataset and split into train/validation
# -----------------------------
dataset = PGJanetSequenceDataset('for_DPD.mat', seq_len=seq_len)
n_val = int(len(dataset) * val_ratio)
n_train = len(dataset) - n_val
train_set, val_set = random_split(dataset, [n_train, n_val])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

# -----------------------------
# 1. Pre-train PG-JANET model
# -----------------------------
print('Pre-training PG-JANET model...')
train_model = TrainModel(seq_len=seq_len, hidden_size=hidden_size, n_epochs=n_epochs, batch_size=batch_size, mat_path='for_DPD.mat')
train_model.train()
model = train_model.model

# -----------------------------
# 2. ILC Integration (Fig. 5)
# -----------------------------
print('Starting ILC integration...')
mat = loadmat('for_DPD.mat', squeeze_me=True)
X_full = mat['TX1_BB']  # Complex baseband input
Y_full = mat['TX1_SISO']  # Distorted output

# Normalize input as in training
X_full_abs = np.abs(X_full).astype(np.float32)
X_full_abs_norm = X_full_abs / np.max(X_full_abs)
X_full_phase = np.angle(X_full).astype(np.float32)

# Initial predistorted input is the original input
X_ilc = X_full.copy()

for ilc_iter in range(ilc_iterations):
    print(f'ILC Iteration {ilc_iter+1}/{ilc_iterations}')
    # 1. Pass current input through PA (simulate by using Y_full as measured output)
    # In real system, you would send X_ilc to PA and measure output
    # Here, we simulate by using Y_full (or you can use a function to simulate PA)
    Y_measured = Y_full  # [N]
    # 2. Calculate error (desired - measured) and nMSE
    error = X_full - Y_measured  # [N], complex
    nMSE = 10 * np.log10(np.sum(np.abs(error) ** 2) / np.sum(np.abs(X_full) ** 2))
    print(f"nMSE at ILC iteration {ilc_iter+1}: {nMSE:.4f} dB")

    # 3. Update predistorted input using PG-JANET model
    # Prepare amplitude and phase for current input
    X_ilc_abs = np.abs(X_ilc).astype(np.float32)
    X_ilc_abs_norm = X_ilc_abs / np.max(X_full_abs)  # Normalize with original max
    X_ilc_phase = np.angle(X_ilc).astype(np.float32)
    X_ilc_abs_tensor = torch.from_numpy(X_ilc_abs_norm).unsqueeze(0).unsqueeze(-1)
    X_ilc_phase_tensor = torch.from_numpy(X_ilc_phase).unsqueeze(0).unsqueeze(-1)
    with torch.no_grad():
        model_output_tensor = model(X_ilc_abs_tensor, X_ilc_phase_tensor)  # [1, N, 2]
    model_output_np = model_output_tensor.squeeze(0).cpu().numpy()  # [N, 2]
    model_output_complex = model_output_np[:, 0] + 1j * model_output_np[:, 1]

    # 4. Update input for next iteration (ILC update rule)
    # X_ilc_new = X_ilc + learning_rate * error (simple ILC update)
    # You can tune learning_rate for stability
    learning_rate = 0.7
    X_ilc = X_ilc + learning_rate * error

# After ILC, X_ilc is the refined predistorted input
X_ilc_abs = np.abs(X_ilc).astype(np.float32)
X_ilc_abs_norm = X_ilc_abs / np.max(X_full_abs)
X_ilc_phase = np.angle(X_ilc).astype(np.float32)

# -----------------------------
# 3. DLA Fine-tuning (using ILC-refined data)
# -----------------------------
print('Starting DLA fine-tuning...')
# Create a new dataset using the ILC-refined input as input, and original X_full as target
class DLARefinedDataset(torch.utils.data.Dataset):
    def __init__(self, X_ilc, X_target, seq_len=12):
        amplitudes = np.abs(X_ilc).astype(np.float32)
        amplitudes /= np.max(amplitudes)
        phases = np.angle(X_ilc).astype(np.float32)
        targets = np.stack([X_target.real, X_target.imag], axis=-1).astype(np.float32)
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

dla_dataset = DLARefinedDataset(X_ilc, X_full, seq_len=seq_len)
dla_loader = DataLoader(dla_dataset, batch_size=batch_size, shuffle=True)

# Fine-tune the model
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_function = nn.MSELoss()
print('Fine-tuning PG-JANET model on DLA-refined data...')
for epoch in range(fine_tune_epochs):
    model.train()
    epoch_loss = 0.0
    for batch_x_abs, batch_theta, batch_targets in dla_loader:
        outputs = model(batch_x_abs, batch_theta)
        loss = loss_function(outputs, batch_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= len(dla_loader)
    print(f"DLA Fine-tune Epoch {epoch+1}/{fine_tune_epochs}, Loss: {epoch_loss:.6f}")
print('DLA fine-tuning complete.')

# Save the final model
torch.save(model.state_dict(), 'pg_janet_rnn_real_ilc_dla.pth')
print('Final model saved as pg_janet_rnn_real_ilc_dla.pth')

# Optionally, you can add evaluation/plotting code as in your previous scripts.
