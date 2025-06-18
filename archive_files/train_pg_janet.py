import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from torch.utils.data import DataLoader
from scipy.io import loadmat
from pg_janet import PGJanetRNN

#############################3
def process_signal(signal):
    signal_conj = np.conj(signal)
    spectrum = np.fft.fftshift(np.fft.fft(signal_conj))
    spectrum_db = 10 * np.log10(np.abs(spectrum) + 1e-12) - 46.3
    smoothed = np.convolve(spectrum_db, np.ones(100)/100, mode='same')
    return smoothed

def plot_frequency_spectrum_from_complex_array(signal1, signal2, signal3, freq_range=128, label1='Signal 1', label2='Signal 2', label3 = 'Signal 3', title='Spectrum'):
    """
    מציג ספקטרום של אות מרוכב 1D כמו TX1_BB או TX1_SISO.
    """
    spectrum1 = process_signal(signal1)
    spectrum2 = process_signal(signal2)
    spectrum3 = process_signal(signal3)

    f = np.linspace(-freq_range / 2, freq_range / 2, len(spectrum1))

    plt.figure(figsize=(10, 5))
    plt.plot(f, spectrum1, color='black', linewidth=2, label=label1)
    plt.plot(f, spectrum2, color='red', linewidth=2, label=label2)
    plt.plot(f, spectrum3, color='blue', linewidth=2, label=label3)
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude [dB]')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

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
        X_m = np.mean(X)
        X_s = np.std(X)
        Y_m = np.mean(X)
        Y_s = np.std(Y)
        X = (X-X_m)/X_s
        Y = (Y-Y_m)/Y_s
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


# # === Plot the freq domain ===

# reconstructed Y

reconstructed_targets = dataset.target_seq[:, 0, :]  #  Take the first element from each sequence
Y_reconstructed = reconstructed_targets[:, 0] + 1j * reconstructed_targets[:, 1] # Build complex numbers
last_few = dataset.target_seq[-1, 1:, :] # Take the last sequence
last_few_complex = last_few[:, 0] + 1j * last_few[:, 1] 
Y_full_reconstructed = np.concatenate([Y_reconstructed, last_few_complex])

# reconstructed X

x_abs_seq = dataset.x_abs_seq  # shape: [num_sequences, seq_len]
theta_seq = dataset.theta_seq  # shape: [num_sequences, seq_len]

# ניקח את האיבר הראשון מכל רצף
x_abs_first = x_abs_seq[:, 0]       # shape: [num_sequences]
theta_first = theta_seq[:, 0]       # shape: [num_sequences]
last_few = dataset.x_abs_seq[-1, 1:] # Take the last sequence
X_abs_full = np.concatenate([x_abs_first, last_few])

last_few = dataset.theta_seq[-1, 1:] # Take the last sequence
theta_full = np.concatenate([theta_first, last_few])

reconstructed_signal = X_abs_full * np.exp(1j * theta_full)  # shape: [num_sequences]


all_outputs = []

model.eval()
with torch.no_grad():
    for batch_x_abs, batch_theta, batch_targets in loader:
        outputs = model(batch_x_abs, batch_theta)
        # print(outputs.size())
        all_outputs.append(outputs.cpu().numpy())

# Concatenate all outputs into a single array
all_outputs_np = np.concatenate(all_outputs, axis=0)  # shape: [num_sequences, seq_len, 2]
full_signal = all_outputs_np.reshape(-1, 2)  # shape: [total_samples, 2]

# Construct a complex signal from real and imaginary parts
full_signal_complex = full_signal[:, 0] + 1j * full_signal[:, 1]
print(full_signal_complex.shape)

num_total = len(dataset.target_seq) + seq_len - 1  # -> reconstruct original N
full_signal_complex = np.zeros((num_total,), dtype=np.complex64)
count = np.zeros((num_total,), dtype=np.float32)

for i in range(len(all_outputs_np)):
    for j in range(seq_len):
        idx = i + j
        real = all_outputs_np[i, j, 0]
        imag = all_outputs_np[i, j, 1]
        full_signal_complex[idx] += real + 1j * imag
        count[idx] += 1

# Average overlapping samples
count[count == 0] = 1  # avoid division by zero at the edges
full_signal_complex /= count

plot_frequency_spectrum_from_complex_array(reconstructed_signal, Y_full_reconstructed, full_signal_complex, freq_range=128,
                                           label1='X', label2='Y', label3='Model Output', title='Spectrum')

