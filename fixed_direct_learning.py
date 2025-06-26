import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.io import loadmat
from fixed_PG_JANET_training import PGJanetRNN, max_norm

# --- Load and preprocess data (direct model: BB -> SISO) ---
mat_path = 'for_DPD.mat'
mat = loadmat(mat_path, squeeze_me=True)
X = mat['TX1_BB']  # Clean input
Y = mat['TX1_SISO']  # Distorted output

x_abs = np.abs(X).astype(np.float32)
x_theta = np.angle(X).astype(np.float32)
y_abs = np.abs(Y).astype(np.float32)
y_theta = np.angle(Y).astype(np.float32)

x_abs_max = np.max(np.abs(x_abs))
y_abs_max = np.max(np.abs(y_abs))

x_abs_norm = max_norm(x_abs, x_abs_max)
y_abs_norm = max_norm(y_abs, y_abs_max)

Y_norm = y_abs_norm * np.exp(1j * y_theta)
y_real_norm = np.real(Y_norm)
y_imag_norm = np.imag(Y_norm)
targets = np.stack([y_real_norm, y_imag_norm], axis=-1).astype(np.float32)

# --- Prepare tensors ---
seq_len = 10
N = len(x_abs_norm)
num_seqs = N // seq_len

x_abs_seq = x_abs_norm[:num_seqs * seq_len].reshape(num_seqs, seq_len, 1)
x_theta_seq = x_theta[:num_seqs * seq_len].reshape(num_seqs, seq_len, 1)
targets_seq = targets[:num_seqs * seq_len].reshape(num_seqs, seq_len, 2)

x_abs_seq_torch = torch.from_numpy(x_abs_seq).float()
x_theta_seq_torch = torch.from_numpy(x_theta_seq).float()
targets_seq_torch = torch.from_numpy(targets_seq).float()

# --- Load direct model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PGJanetRNN(hidden_size=64).to(device)
model.load_state_dict(torch.load("pg_janet_rnn_max.pth", map_location=device))
model.eval()

x_abs_seq_torch = x_abs_seq_torch.to(device)
x_theta_seq_torch = x_theta_seq_torch.to(device)

# --- Evaluate model ---
with torch.no_grad():
    outputs = model(x_abs_seq_torch, x_theta_seq_torch)
    outputs_np = outputs.cpu().numpy().reshape(-1, 2)
    x_abs_flat = x_abs_seq_torch.cpu().numpy().reshape(-1)
    x_theta_flat = x_theta_seq_torch.cpu().numpy().reshape(-1)
    targets_flat = targets_seq_torch.cpu().numpy().reshape(-1, 2)

# --- AM/AM Plot ---
y_pred_magnitude = np.sqrt(outputs_np[:, 0]**2 + outputs_np[:, 1]**2)
y_real_magnitude = np.sqrt(targets_flat[:, 0]**2 + targets_flat[:, 1]**2)

x_abs_plot = x_abs_flat * x_abs_max
y_pred_plot = y_pred_magnitude * y_abs_max
y_real_plot = y_real_magnitude * y_abs_max

plt.figure(figsize=(8, 6))
plt.scatter(x_abs_plot, y_real_plot, s=1, color='blue', label='Original (No DPD)')
plt.scatter(x_abs_plot, y_pred_plot, s=1, color='red', label='With PG-JANET')
plt.xlabel("Input Magnitude")
plt.ylabel("Output Magnitude")
plt.title("AM/AM Comparison: Original vs PG-JANET (Direct, Denormalized)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('pg_janet_am_am_comparison.png')
plt.show()

# --- AM/PM ---
x_magnitude_list = []
y_pred_phase_list = []
y_real_phase_list = []

with torch.no_grad():
    for i in range(x_abs_seq_torch.shape[0]):
        x_mag = x_abs_seq_torch[i].cpu().numpy().flatten()
        x_theta = x_theta_seq_torch[i].cpu().numpy().flatten()
        y_pred = outputs[i].cpu().numpy()
        y_real = targets_seq_torch[i].cpu().numpy()

        y_pred_c = y_pred[:, 0] + 1j * y_pred[:, 1]
        y_real_c = y_real[:, 0] + 1j * y_real[:, 1]

        x_theta_unwrap = np.unwrap(x_theta)
        y_pred_phase = np.unwrap(np.angle(y_pred_c))
        y_real_phase = np.unwrap(np.angle(y_real_c))

        ampm_pred_seq = y_pred_phase - x_theta_unwrap
        ampm_real_seq = y_real_phase - x_theta_unwrap

        x_magnitude_list.append(x_mag)
        y_pred_phase_list.append(ampm_pred_seq)
        y_real_phase_list.append(ampm_real_seq)

x_magnitude = np.concatenate(x_magnitude_list) * x_abs_max
am_pm_pred = np.concatenate(y_pred_phase_list)
am_pm_real = np.concatenate(y_real_phase_list)

center = np.mean(am_pm_real)
am_pm_real -= center
am_pm_pred -= center

plt.figure(figsize=(8, 6))
plt.scatter(x_magnitude, am_pm_real, s=1, color='blue', label='Original (No DPD)')
plt.scatter(x_magnitude, am_pm_pred, s=1, color='red', label='With PG-JANET')
plt.xlabel("Input Magnitude")
plt.ylabel("Phase Shift (radians)")
plt.title("AM/PM Comparison: Original vs PG-JANET (Per-sequence unwrap)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("pg_janet_am_pm_fixed_per_sequence.png")
plt.show()

# --- Spectrum Plot ---
predicted_signal = y_pred_plot * np.exp(1j * am_pm_pred + 1j * x_theta_flat)
real_signal = y_real_plot * np.exp(1j * am_pm_real + 1j * x_theta_flat)

from scipy.fft import fft, fftshift
from scipy.signal import savgol_filter

def plot_spectrum(sig, label, color):
    sig_conj = np.conj(sig)
    spectrum = fftshift(fft(sig_conj))
    spectrum_db = 20 * np.log10(np.abs(spectrum) + 1e-12) - 46.3
    spectrum_db_smooth = savgol_filter(spectrum_db, 101, 3)
    N = len(sig)
    f = np.linspace(-0.5, 0.5, N)
    plt.plot(f, spectrum_db_smooth, color=color, label=label, linewidth=2)

plt.figure(figsize=(10, 6))
plot_spectrum(x_abs_flat * np.exp(1j * x_theta_flat), 'Original Input', 'black')
plot_spectrum(real_signal, 'PA Output', 'blue')
plot_spectrum(predicted_signal, 'PG-JANET Output', 'red')
plt.title("Spectrum Comparison (FFT, Smoothed, Direct)")
plt.xlabel("Normalized Frequency")
plt.ylabel("Magnitude (dB)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("pg_janet_spectrum_comparison.png")
plt.show()
