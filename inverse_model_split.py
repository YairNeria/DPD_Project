import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from train_pg_janet_class import PGJanetSequenceDataset, TrainModel

# --- Hyperparameters ---
seq_len = 3
hidden_size = 32
n_epochs = 30
batch_size = 64
val_ratio = 0.3
learning_rate = 5e-3

# --- Dataset: Inverse modeling (input = distorted, target = original) ---
def std_norm(x, mean=None, std=None):
    if mean is None: mean = x.mean()
    if std is None: std = x.std()
    return (x - mean) / (std + 1e-8)

dataset = PGJanetSequenceDataset('for_DPD.mat', seq_len=seq_len, invert=True, Norm_func=std_norm)
n_val = int(len(dataset) * val_ratio)
n_train = len(dataset) - n_val
train_set, val_set = random_split(dataset, [n_train, n_val])
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

# --- Train the model ---
model = TrainModel(seq_len=seq_len, hidden_size=hidden_size, n_epochs=n_epochs,
                   batch_size=batch_size, mat_path='for_DPD.mat', learning_rate=learning_rate)
model.train()

# --- Evaluation ---
input_mag_all, pred_mag_all, targ_mag_all = [], [], []
am_pm_diff_all, am_pm_nodp_all = [], []

with torch.no_grad():
    for batch_x_abs, batch_theta, batch_targets in val_loader:
        # Forward pass
        pred = model.model(batch_x_abs, batch_theta)

        # Reconstruct complex target (original, undistorted)
        targ = batch_targets.squeeze(0).cpu().numpy()  # shape: (seq_len, 2)
        x_true_complex = targ[:, 0] + 1j * targ[:, 1]  # true original (x)

        # Model output (predicted original)
        pred = pred.squeeze(0).cpu().numpy()  # shape: (seq_len, 2)
        x_pred_complex = pred[:, 0] + 1j * pred[:, 1]  # predicted original (x_hat)

        # Reconstruct complex input (distorted signal)
        batch_x_abs = batch_x_abs.squeeze(0).cpu().numpy().flatten()
        batch_theta = batch_theta.squeeze(0).cpu().numpy().flatten()
        input_real = batch_x_abs * np.cos(batch_theta)
        input_imag = batch_x_abs * np.sin(batch_theta)
        y_complex = input_real + 1j * input_imag  # y: distorted signal

        # Get magnitudes
        y_mag = np.abs(y_complex)
        x_pred_mag = np.abs(x_pred_complex)
        x_true_mag = np.abs(x_true_complex)

        # Get phases (for AM/PM)
        pred_phase = np.angle(x_pred_complex, deg=True)
        true_phase = np.angle(x_true_complex, deg=True)
        y_phase = np.angle(y_complex, deg=True)

        # AM/PM phase differences
        phase_diff = true_phase - pred_phase          # AM/PM with DPD
        nodp_phase_diff = true_phase - y_phase        # AM/PM without DPD

        # Filter out near-zero distorted magnitudes
        valid = y_mag > 0.05
        input_mag_all.append(y_mag[valid])
        pred_mag_all.append(x_pred_mag[valid])
        targ_mag_all.append(x_true_mag[valid])
        am_pm_diff_all.append(phase_diff[valid])
        am_pm_nodp_all.append(nodp_phase_diff[valid])

# --- Stack and Normalize ---
y_mag = np.concatenate(input_mag_all)
x_pred_mag = np.concatenate(pred_mag_all)
x_true_mag = np.concatenate(targ_mag_all)
phase_dpd = np.concatenate(am_pm_diff_all)
phase_nodp = np.concatenate(am_pm_nodp_all)

# Shared normalization factor (based on true original signal)
norm_factor = np.max(x_true_mag)
x_norm = x_true_mag / norm_factor         # ground truth x (x-axis)
x_hat_norm = x_pred_mag / norm_factor     # predicted x_hat
y_norm = y_mag / norm_factor              # distorted input y

# --- Plot ---

# --- AM/AM Plot ---
fig_amam, ax_amam = plt.subplots(figsize=(8, 6))
ax_amam.scatter(x_norm, y_norm, s=2, alpha=0.3, color='red', label='AM/AM without DPD (x → y)')
ax_amam.scatter(x_norm, x_hat_norm, s=2, alpha=0.3, color='black', label='AM/AM with PG-JANET (x → x̂)')
ax_amam.plot(np.linspace(0, 1, 200), np.linspace(0, 1, 200), 'k--', label='Ideal')
ax_amam.set_xlabel("Normalized True Original Amplitude (x)")
ax_amam.set_ylabel("Normalized Amplitude (x̂ or y)")
ax_amam.set_xlim(0, 1)
ax_amam.set_ylim(0, 1)
ax_amam.legend(loc='upper left')
ax_amam.set_title("AM/AM Characteristics with PG-JANET (Inverse Model, y → x)")
ax_amam.grid(True)
plt.tight_layout()
fig_amam.savefig("AM_AM_PG_JANET_inverse.png")

# --- AM/PM Plot ---
fig_ampm, ax_ampm = plt.subplots(figsize=(8, 6))
ax_ampm.scatter(x_norm, phase_nodp, s=2, alpha=0.2, color='blue', label='AM/PM without DPD (x − y)')
ax_ampm.scatter(x_norm, phase_dpd, s=2, alpha=0.2, color='magenta', label='AM/PM with PG-JANET (x − x̂)')
ax_ampm.set_xlabel("Normalized True Original Amplitude (x)")
ax_ampm.set_ylabel("Phase Difference (Degree)")
ax_ampm.set_ylim(-180, 180)
ax_ampm.legend(loc='upper left')
ax_ampm.set_title("AM/PM Characteristics with PG-JANET (Inverse Model, y → x)")
ax_ampm.grid(True)
plt.tight_layout()
fig_ampm.savefig("AM_PM_PG_JANET_inverse.png")

# Optionally, show both plots
# plt.show()
