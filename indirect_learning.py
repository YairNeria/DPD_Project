import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.io import loadmat
from scipy.fft import fft, fftshift
from scipy.signal import savgol_filter
from train_pg_janet_class import PGJanetRNN
from train_pg_janet_class import state_norm

#PG JANET TRAINED ON U_ILC 
# Load data
mat = loadmat('for_DPD.mat', squeeze_me=True)
X = mat['TX1_BB']
Y = mat['TX1_SISO']

# In indirect learning: input is distorted Y, target is original X
amplitudes = np.abs(Y).astype(np.float32)
phases = np.angle(Y).astype(np.float32)
targets = np.stack([X.real, X.imag], axis=-1).astype(np.float32)

# Use normalization parameters from training

amplitudes_norm = (amplitudes - state_norm['mean_amp']) / state_norm['std_amp']
targets_norm = (targets - state_norm['mean_targets']) / state_norm['std_targets']
mean = np.concatenate([[state_norm['mean_amp']], state_norm['mean_targets']])
std = np.concatenate([[state_norm['std_amp']], state_norm['std_targets']])

# Prepare sequences
seq_len = 10
x_abs_seq, theta_seq = [], []
for i in range(len(amplitudes_norm) - seq_len):
    x_abs_seq.append(amplitudes_norm[i:i+seq_len])
    theta_seq.append(phases[i:i+seq_len])

# Ensure arrays are proper shape before converting to torch tensors
x_abs_seq = np.array(x_abs_seq)  # shape (N, seq_len)
theta_seq = np.array(theta_seq)  # shape (N, seq_len)
x_abs_seq = torch.tensor(x_abs_seq).unsqueeze(-1).float()  # shape (N, seq_len, 1)
theta_seq = torch.tensor(theta_seq).unsqueeze(-1).float()  # shape (N, seq_len, 1)


# Load inverse-trained PG-JANET model
model = PGJanetRNN(hidden_size=64)
model.load_state_dict(torch.load("pg_janet_rnn_inverse.pth", map_location=torch.device('cpu')))
model.eval()

# Predict
with torch.no_grad():
    y_pred_norm = model(x_abs_seq, theta_seq).numpy()
    #it is actually x because we are predicting the original signal from the distorted one

# Denormalize
y_pred = y_pred_norm * std[1:] + mean[1:]
y_pred_complex = y_pred[:, -1, 0] + 1j * y_pred[:, -1, 1]
x_used = Y[seq_len:len(y_pred_complex)+seq_len]
targets_used = X[seq_len:len(y_pred_complex)+seq_len]

x_seq = np.array([X[i:i+seq_len] for i in range(len(X) - seq_len)])
x_last = x_seq[:, -1]
# -------------------------------
# AM/AM Plot
# -------------------------------
input_mag = np.abs(x_used)
output_mag = np.abs(targets_used)
input_mag_pred = np.abs(x_used)
output_mag_pred = np.abs(y_pred_complex)

input_mag_norm = input_mag / np.max(input_mag)
output_mag_norm = output_mag / np.max(input_mag)
input_mag_pred /= np.max(input_mag)
output_mag_pred /= np.max(input_mag)

plt.figure(figsize=(8, 6))
plt.scatter(input_mag_norm, output_mag_norm, s=1, color='blue', label='Original (Before Inverse)')
plt.scatter(input_mag_pred, output_mag_pred, s=1, color='red', label='After PG-JANET Inverse')
plt.xlabel("Normalized Input Magnitude")
plt.ylabel("Normalized Output Magnitude")
plt.title("AM/AM Comparison: Indirect Learning (Inverse)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('pg_janet_inverse_am_am_comparison.png')
plt.show()

# -------------------------------
# AM/PM Plot
# -------------------------------


input_phase = np.angle(Y)
output_phase = np.angle(X)
am_pm_original = np.unwrap(output_phase - input_phase)

input_phase_pred = np.angle(x_last)
output_phase_pred = np.angle(y_pred_complex)
am_pm_pred = np.unwrap(output_phase_pred - input_phase_pred)

input_mag = np.abs(Y)
input_mag_pred = np.abs(x_last)
input_mag /= np.max(input_mag)
input_mag_pred /= np.max(input_mag)

am_pm_pred -= np.mean(am_pm_pred)
am_pm_original = am_pm_original[seq_len:]
am_pm_original -= np.mean(am_pm_original)

plt.figure(figsize=(8, 6))
plt.scatter(input_mag[seq_len:], am_pm_original, s=1, color='blue', label='Original')
plt.scatter(input_mag_pred, am_pm_pred, s=1, color='red', label='PG-JANET Inverse')
plt.xlabel("Normalized Input Magnitude")
plt.ylabel("Phase Shift (radians)")
plt.title("AM/PM Comparison: Indirect Learning (Inverse)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('pg_janet_inverse_am_pm_comparison_fixed.png')
plt.show()

# -------------------------------
# Spectrum Plot
# -------------------------------
# todo - display from -90 to 10 dB and not any lower 

def plot_spectrum(sig, label, color):
    sig_conj = np.conj(sig)
    spectrum = fftshift(fft(sig_conj))
    spectrum_db = 20 * np.log10(np.abs(spectrum) + 1e-12) - 46.3
    spectrum_db_smooth = savgol_filter(spectrum_db, 101, 3)
    N = len(sig)
    f = np.linspace(-0.5, 0.5, N)
    plt.plot(f, spectrum_db_smooth, color=color, label=label, linewidth=2)

predicted_signal = np.zeros_like(Y, dtype=np.complex64)
predicted_signal[seq_len:seq_len + len(y_pred_complex)] = y_pred_complex

plt.figure(figsize=(10, 6))
plot_spectrum(Y, 'PA Output (Distorted)', 'blue')
plot_spectrum(X, 'Original Input', 'black')
plot_spectrum(predicted_signal, 'PG-JANET Inverse Output', 'red')
plt.title("Spectrum Comparison (Inverse Learning)")
plt.xlabel("Normalized Frequency")
plt.ylabel("Magnitude (dB)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("pg_janet_inverse_spectrum_comparison.png")
plt.show()
