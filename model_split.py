
# Import required modules and classes
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from train_pg_janet_class import PGJanetSequenceDataset, TrainModel

# -----------------------------
# Hyperparameters
# -----------------------------
seq_len = 12           # Length of each input sequence
hidden_size = 20      # Hidden size for the RNN
n_epochs = 30          # Number of training epochs
batch_size = 64        # Batch size for DataLoader
val_ratio = 0.2        # Fraction of data to use for validation

# -----------------------------
# Load dataset and split into train/validation
# -----------------------------
dataset = PGJanetSequenceDataset('for_DPD.mat', seq_len=seq_len)
n_val = int(len(dataset) * val_ratio)
n_train = len(dataset) - n_val
train_set, val_set = random_split(dataset, [n_train, n_val])

# Create DataLoaders for training and validation
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)



# -----------------------------
# Use TrainModel class for training
# -----------------------------
print('Starting training using TrainModel class...')
# Create a TrainModel instance with the same hyperparameters and training set
from train_pg_janet_class import TrainModel
train_model = TrainModel(seq_len=seq_len, hidden_size=hidden_size, n_epochs=n_epochs, batch_size=batch_size, mat_path='for_DPD.mat')
train_model.train()
# model = train_model.model  # Use the trained model for validation

# # -----------------------------
# # Validation: get predictions and real values
# # -----------------------------
# train_model.model.eval()
# y_pred_list = []
# y_real_list = []
# with torch.no_grad():
#     for batch_x_abs, batch_theta, batch_targets in val_loader:
#         # Forward pass for validation
#         outputs = train_model.model(batch_x_abs, batch_theta)  # [1, seq_len, 2]
#         y_pred_list.append(outputs.squeeze(0).cpu().numpy())
#         y_real_list.append(batch_targets.squeeze(0).cpu().numpy())

# # Concatenate all sequences for plotting
# y_pred = np.concatenate(y_pred_list, axis=0)  # [N_val*seq_len, 2]
# y_real = np.concatenate(y_real_list, axis=0)  # [N_val*seq_len, 2]

# # -----------------------------

# # Plot real and predicted signals (I and Q)
# # -----------------------------
# plt.figure(figsize=(12, 6))
# # Plot I component
# plt.subplot(2, 1, 1)
# plt.plot(y_real[:, 0], label='Real I')
# plt.plot(y_pred[:, 0], label='Predicted I', alpha=0.7)
# plt.title('I Component')
# plt.legend()
# # Plot Q component
# plt.subplot(2, 1, 2)
# plt.plot(y_real[:, 1], label='Real Q')
# plt.plot(y_pred[:, 1], label='Predicted Q', alpha=0.7)
# plt.title('Q Component')
# plt.legend()
# plt.xlabel('Sample (real timescale)')
# plt.tight_layout()
# plt.show()
# # -----------------------------
# # Plot: AM/AM characteristics
# # -----------------------------
# # Compute input magnitude (x_abs), predicted output magnitude, and real output magnitude
# x_magnitude_list = []
# y_pred_magnitude_list = []
# y_real_magnitude_list = []

# with torch.no_grad():
#     for batch_x_abs, batch_theta, batch_targets in val_loader:
#         # batch_x_abs: [1, seq_len, 1]
#         # batch_targets: [1, seq_len, 2]
#         outputs = train_model.model(batch_x_abs, batch_theta)  # [1, seq_len, 2]
#         x_magnitude = batch_x_abs.squeeze(0).cpu().numpy().flatten()  # [seq_len]
#         y_pred = outputs.squeeze(0).cpu().numpy()  # [seq_len, 2]
#         y_real = batch_targets.squeeze(0).cpu().numpy()  # [seq_len, 2]
#         y_pred_magnitude = np.sqrt(y_pred[:, 0]**2 + y_pred[:, 1]**2)
#         y_real_magnitude = np.sqrt(y_real[:, 0]**2 + y_real[:, 1]**2)
#         x_magnitude_list.append(x_magnitude)
#         y_pred_magnitude_list.append(y_pred_magnitude)
#         y_real_magnitude_list.append(y_real_magnitude)

# # Concatenate all sequences
# x_magnitude = np.concatenate(x_magnitude_list, axis=0)
# y_pred_magnitude = np.concatenate(y_pred_magnitude_list, axis=0)
# y_real_magnitude = np.concatenate(y_real_magnitude_list, axis=0)
# # Scatter plot: Predicted output magnitude vs input magnitude (AM/AM)
# plt.figure(figsize=(8, 6))
# plt.scatter(x_magnitude, y_pred_magnitude, alpha=0.5, s=5, label='Predicted')
# plt.scatter(x_magnitude, y_real_magnitude, alpha=0.5, s=5, label='Real')

# # Plot ideal linear AM/AM (output = input)
# x_ideal = np.linspace(x_magnitude.min(), x_magnitude.max(), 200)
# plt.plot(x_ideal, x_ideal, 'k--', linewidth=2, label='Ideal (Linear)')

# plt.xlabel('Input Magnitude (|x|)')
# plt.ylabel('Output Magnitude (|y|)')
# plt.title('AM/AM Characteristics')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # -----------------------------
# # Scatter plot: Predicted output phase vs input phase (AM/PM)   
# # -----------------------------
# # Compute input phase (theta), predicted output phase, and real output phase
# x_phase_list = []   
# y_pred_phase_list = []
# y_real_phase_list = []  

# with torch.no_grad():
#     for batch_x_abs, batch_theta, batch_targets in val_loader:
#         # batch_x_abs: [1, seq_len, 1]
#         # batch_theta: [1, seq_len, 1]
#         # batch_targets: [1, seq_len, 2]
#         outputs = train_model.model(batch_x_abs, batch_theta)  # [1, seq_len, 2]
#         x_phase = batch_theta.squeeze(0).cpu().numpy().flatten()  # [seq_len]
#         y_pred = outputs.squeeze(0).cpu().numpy()  # [seq_len, 2]
#         y_real = batch_targets.squeeze(0).cpu().numpy()  # [seq_len, 2]
#         y_pred_phase = np.angle(y_pred[:, 0] + 1j * y_pred[:, 1])
#         y_real_phase = np.angle(y_real[:, 0] + 1j * y_real[:, 1])
#         x_phase_list.append(x_phase)
#         y_pred_phase_list.append(y_pred_phase)
#         y_real_phase_list.append(y_real_phase)
# # Concatenate all sequences
# x_phase = np.concatenate(x_phase_list, axis=0)
# y_pred_phase = np.concatenate(y_pred_phase_list, axis=0)
# y_real_phase = np.concatenate(y_real_phase_list, axis=0)

# # Scatter plot: Predicted output phase vs input phase (AM/PM)
# plt.figure(figsize=(8, 6))
# plt.scatter(x_phase, y_pred_phase, alpha=0.5, s=5, label='Predicted')
# plt.scatter(x_phase, y_real_phase, alpha=0.5, s=5, label='Real')
# # Plot ideal linear AM/PM (output = input)
# x_ideal_phase = np.linspace(x_phase.min(), x_phase.max(), 200)
# plt.plot(x_ideal_phase, x_ideal_phase, 'k--', linewidth=2, label='Ideal (Linear)')
# plt.xlabel('Input Phase (theta)')
# plt.ylabel('Output Phase (theta)')
# plt.title('AM/PM Characteristics')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# -----------------------------
# Spectrum plot: compare original, distorted, and model output
# -----------------------------
def plot_spectrum(sig, label):
    sig_conj = np.conj(sig)
    spectrum = np.fft.fftshift(np.fft.fft(sig_conj))
    spectrum_db = 10 * np.log10(np.abs(spectrum) + 1e-12) - 46.3
    smoothed = np.convolve(spectrum_db, np.ones(100)/100, mode='same')
    freq = np.linspace(-64, 64, len(sig))
    mask = (freq >= -28) & (freq <= 28)
    plt.plot(freq[mask], smoothed[mask], label=label)
    # plt.plot(freq, smoothed, label=label)

# Load the full input and output signals from the .mat file
import scipy.io
mat = scipy.io.loadmat('for_DPD.mat', squeeze_me=True)
X_full = mat['TX1_BB']  # Complex baseband input
Y_full = mat['TX1_SISO']  # Distorted output

# PG-JANET model output (predistorted):
# We'll use the trained model to generate the output for the full input sequence
# Prepare amplitude and phase for the full input
X_full_abs = np.abs(X_full).astype(np.float32)
X_full_abs_norm = X_full_abs / np.max(X_full_abs)  # Normalize as in training
X_full_phase = np.angle(X_full).astype(np.float32)

# Convert to torch tensors and add batch/seq dims
X_full_abs_tensor = torch.from_numpy(X_full_abs_norm).unsqueeze(0).unsqueeze(-1)  # [1, N, 1]
X_full_phase_tensor = torch.from_numpy(X_full_phase).unsqueeze(0).unsqueeze(-1)   # [1, N, 1]

with torch.no_grad():
    Model_output_full_tensor = train_model.model(X_full_abs_tensor, X_full_phase_tensor)  # [1, N, 2]
Model_output_full_np = Model_output_full_tensor.squeeze(0).cpu().numpy()  # [N, 2]
Model_output_full = Model_output_full_np[:, 0] + 1j * Model_output_full_np[:, 1]

plt.figure(figsize=(10, 6))
plot_spectrum(X_full, "Original Input")
plot_spectrum(Y_full, "PA Output")
plot_spectrum(Model_output_full, "PG-JANET Output")
plt.xlabel("Normalized Frequency")
plt.ylabel("Magnitude (dB)")
plt.legend()
plt.title("Frequency Spectrum Comparison")
plt.tight_layout()
plt.show()
