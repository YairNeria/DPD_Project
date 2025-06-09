
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
hidden_size = 16       # Hidden size for the RNN
n_epochs = 20          # Number of training epochs
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
model = train_model.model  # Use the trained model for validation

# -----------------------------
# Validation: get predictions and real values
# -----------------------------
train_model.model.eval()
y_pred_list = []
y_real_list = []
with torch.no_grad():
    for batch_x_abs, batch_theta, batch_targets in val_loader:
        # Forward pass for validation
        outputs = train_model.model(batch_x_abs, batch_theta)  # [1, seq_len, 2]
        y_pred_list.append(outputs.squeeze(0).cpu().numpy())
        y_real_list.append(batch_targets.squeeze(0).cpu().numpy())

# Concatenate all sequences for plotting
y_pred = np.concatenate(y_pred_list, axis=0)  # [N_val*seq_len, 2]
y_real = np.concatenate(y_real_list, axis=0)  # [N_val*seq_len, 2]

# -----------------------------

# Plot real and predicted signals (I and Q)
# -----------------------------
plt.figure(figsize=(12, 6))
# Plot I component
plt.subplot(2, 1, 1)
plt.plot(y_real[:, 0], label='Real I')
plt.plot(y_pred[:, 0], label='Predicted I', alpha=0.7)
plt.title('I Component')
plt.legend()
# Plot Q component
plt.subplot(2, 1, 2)
plt.plot(y_real[:, 1], label='Real Q')
plt.plot(y_pred[:, 1], label='Predicted Q', alpha=0.7)
plt.title('Q Component')
plt.legend()
plt.xlabel('Sample (real timescale)')
plt.tight_layout()
plt.show()

# -----------------------------
# Plot: Predicted amplitude vs Real amplitude (scatter plot)
# -----------------------------
real_amplitude = np.sqrt(y_real[:, 0]**2 + y_real[:, 1]**2)
pred_amplitude = np.sqrt(y_pred[:, 0]**2 + y_pred[:, 1]**2)

plt.figure(figsize=(6, 6))
plt.scatter(real_amplitude, pred_amplitude, alpha=0.5, s=5)
plt.xlabel('Real Amplitude')
plt.ylabel('Predicted Amplitude')
plt.title('Predicted vs Real Amplitude')
plt.grid(True)
plt.plot([real_amplitude.min(), real_amplitude.max()], [real_amplitude.min(), real_amplitude.max()], 'r--', label='Ideal')
plt.legend()
plt.tight_layout()
plt.show()

y_pred_list = []
y_real_list = []


# (Removed duplicate and unnecessary code after line 80; only the TrainModel-based workflow and validation/plotting remain)