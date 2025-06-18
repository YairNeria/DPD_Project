import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.io import loadmat
from train_pg_janet_class import PGJanetRNN

# Load data
mat = loadmat('for_DPD.mat', squeeze_me=True)
X = mat['TX1_BB']
Y = mat['TX1_SISO']

# Preprocess
amplitudes = np.abs(X).astype(np.float32)
phases = np.angle(X).astype(np.float32)
targets = np.stack([Y.real, Y.imag], axis=-1).astype(np.float32)

# Normalize using same stats for input & output
combined = np.concatenate([amplitudes.reshape(-1, 1), targets], axis=1)
mean = combined.mean(axis=0)
std = combined.std(axis=0)

amplitudes_norm = (amplitudes - mean[0]) / (std[0] + 1e-8)
targets_norm = (targets - mean[1:]) / (std[1:] + 1e-8)
# Create sequence inputs for PG-JANET
seq_len = 3
x_abs_seq, theta_seq = [], []
for i in range(len(amplitudes_norm) - seq_len):
    x_abs_seq.append(amplitudes_norm[i:i+seq_len])
    theta_seq.append(phases[i:i+seq_len])

x_abs_seq = torch.tensor(x_abs_seq).unsqueeze(-1)  # (N, seq_len, 1)
theta_seq = torch.tensor(theta_seq).unsqueeze(-1)  # (N, seq_len, 1)

# Load model
model = PGJanetRNN(hidden_size=32)
model.load_state_dict(torch.load("pg_janet_rnn.pth", map_location=torch.device('cpu')))
model.eval()

# Predict
with torch.no_grad():
    y_pred_norm = model(x_abs_seq, theta_seq).numpy()



# Denormalize
y_pred = y_pred_norm * std[1:] + mean[1:]
y_pred_complex = y_pred[:, -1, 0] + 1j * y_pred[:, -1, 1]
x_used = X[seq_len:len(y_pred_complex)+seq_len]


# -------------------------------
# AM/AM Plot
# -------------------------------
input_mag = np.abs(X)
output_mag = np.abs(Y)
input_mag_pred = np.abs(x_used)
output_mag_pred = np.abs(y_pred_complex)

input_mag_norm = input_mag / np.max(input_mag)
output_mag_norm = output_mag / np.max(input_mag)
input_mag_pred /= np.max(input_mag)
output_mag_pred /= np.max(input_mag)

plt.figure(figsize=(8, 6))
plt.scatter(input_mag_norm, output_mag_norm, s=1, color='blue', label='Original (No DPD)')
plt.scatter(input_mag_pred, output_mag_pred, s=1, color='red', label='With PG-JANET')
plt.xlabel("Normalized Input Magnitude")
plt.ylabel("Normalized Output Magnitude")
plt.title("AM/AM Comparison: Original vs PG-JANET")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('pg_janet_am_am_comparison.png')
plt.show()

# -------------------------------
# AM/PM Plot
# -------------------------------
# --- FIXED AM/PM Section ---
# Get last inputs to match prediction step
x_seq = np.array([X[i:i+seq_len] for i in range(len(X) - seq_len)])
x_last = x_seq[:, -1]

# Phase shift = angle(Y) - angle(X_last)
input_phase = np.angle(X)
output_phase = np.angle(Y)
am_pm_original = np.unwrap(output_phase - input_phase)

input_phase_pred = np.angle(x_last)
output_phase_pred = np.angle(y_pred_complex)
am_pm_pred = np.unwrap(output_phase_pred - input_phase_pred)

# Normalize magnitudes
input_mag = np.abs(X)
input_mag_pred = np.abs(x_last)
input_mag /= np.max(input_mag)
input_mag_pred /= np.max(input_mag)

#normalize phase shifts
am_pm_pred -= np.mean(am_pm_pred)
am_pm_original = am_pm_original[seq_len:]
am_pm_original -= np.mean(am_pm_original)


# Plot
plt.figure(figsize=(8, 6))
plt.scatter(input_mag[seq_len:], am_pm_original, s=1, color='blue', label='Original (No DPD)')
plt.scatter(input_mag_pred, am_pm_pred, s=1, color='red', label='With PG-JANET')
plt.xlabel("Normalized Input Magnitude")
plt.ylabel("Phase Shift (radians)")
plt.title("AM/PM Comparison: Original vs PG-JANET (Aligned)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('pg_janet_am_pm_comparison_fixed.png')
plt.show()
