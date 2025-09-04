import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.io import loadmat, savemat
from scipy.fft import fft, fftshift
from scipy.signal import savgol_filter

# -----------------------------
# Try to import PGJanetRNN from common locations
# -----------------------------
PGJANET_IMPORT_ERROR = (
    "Could not import PGJanetRNN. Make sure one of these exists:\n"
    " - from pg_janet import PGJanetRNN\n"
    " - from archive_files.train_pg_janet_class import PGJanetRNN\n"
    "Adjust the import below accordingly."
)
try:
    from pg_janet import PGJanetRNN  # preferred (matches your training script)
except Exception:
    try:
        from archive_files.train_pg_janet_class import PGJanetRNN  # fallback path
    except Exception:
        raise ImportError(PGJANET_IMPORT_ERROR)

# -----------------------------
# Helpers
# -----------------------------
def compute_global_rms(arr: np.ndarray, epsilon: float = 1e-8) -> float:
    """Compute a single RMS across the whole array."""
    return float(np.sqrt(np.mean(np.abs(arr) ** 2) + epsilon))

def to_complex(re_im: np.ndarray) -> np.ndarray:
    """(N,2) -> complex vector"""
    return re_im[..., 0] + 1j * re_im[..., 1]

def from_mag_phase(mag: np.ndarray, phase: np.ndarray) -> np.ndarray:
    return mag * np.exp(1j * phase)

def odd_leq(n: int) -> int:
    """Largest odd integer <= n."""
    return n if n % 2 == 1 else n - 1

def sg_smooth(x: np.ndarray, window_try: int = 101, poly: int = 3) -> np.ndarray:
    """
    Savitzky–Golay smoothing that adapts the window:
    - window_length must be odd
    - 5 <= window_length <= len(x)-1
    Falls back to no smoothing if the array is too short.
    """
    N = len(x)
    if N < 7:  # too short to smooth meaningfully
        return x
    wl = min(window_try, N - 1)
    wl = max(5, odd_leq(wl))
    p = min(poly, wl - 2) if wl >= 7 else 2
    return savgol_filter(x, wl, p)

# -----------------------------
# Load and prepare data
# -----------------------------
mat_path = 'DPD_signal.mat'
if not os.path.exists(mat_path):
    raise FileNotFoundError(
        f"Could not find '{mat_path}'. Put your MAT file next to this script or update mat_path."
    )

mat = loadmat(mat_path, squeeze_me=True)

# Indirect learning setting:
# X = original (clean) input; Y = distorted target (what PA produced)
try:
    X = mat['sig_out']     # distorted baseband (Y)
    Y = mat['sig_in']    # complex clean baseband (X)
except KeyError as e:
    raise KeyError(
        f"Missing expected keys in {mat_path}: {e}. "
        "Expected 'sig_in' and 'sig_out'."
    )

# Base decompositions
x_abs = np.abs(X).astype(np.float32)         # (N,)
x_theta = np.angle(X).astype(np.float32)     # (N,)

y_abs = np.abs(Y).astype(np.float32)         # (N,)
y_theta = np.angle(Y).astype(np.float32)     # (N,)

# --------- GLOBAL RMS normalization (same convention as training) ----------
x_rms = compute_global_rms(x_abs)            # scalar
y_rms = compute_global_rms(y_abs)            # scalar

x_abs_norm = (x_abs / (x_rms + 1e-8)).astype(np.float32)
y_abs_norm = (y_abs / (y_rms + 1e-8)).astype(np.float32)

# Targets in normalized (real, imag)
Y_norm = y_abs_norm * np.exp(1j * y_theta)
targets = np.stack([np.real(Y_norm), np.imag(Y_norm)], axis=-1).astype(np.float32)  # (N,2)

# -----------------------------
# Build sequences for the model
# -----------------------------
seq_len = 16
N = len(x_abs_norm)
num_seqs = N // seq_len

# Trim to full batches of seq_len
cut = num_seqs * seq_len
x_abs_seq_np = x_abs_norm[:cut].reshape(num_seqs, seq_len, 1)
x_theta_seq_np = x_theta[:cut].reshape(num_seqs, seq_len, 1)
targets_seq_np = targets[:cut].reshape(num_seqs, seq_len, 2)

x_abs_seq_t = torch.from_numpy(x_abs_seq_np).float()    # (B,T,1)
x_theta_seq_t = torch.from_numpy(x_theta_seq_np).float()
targets_seq_t = torch.from_numpy(targets_seq_np).float()

# -----------------------------
# Load inverse PG-JANET model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PGJanetRNN(hidden_size=32).to(device)

ckpt_path = "pg_janet_inverse_global_rms.pth"
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(
        f"Checkpoint '{ckpt_path}' not found. Make sure you trained & saved with global RMS."
    )

state = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state)
model.eval()

x_abs_seq_bt = x_abs_seq_t.to(device).squeeze(-1)   # (B,T)
x_theta_seq_bt = x_theta_seq_t.to(device).squeeze(-1)

# -----------------------------
# Evaluate
# -----------------------------
with torch.no_grad():
    outputs_bt = model(x_abs_seq_bt, x_theta_seq_bt)     # (B,T,2), normalized scale
    outputs_np = outputs_bt.cpu().numpy().reshape(-1, 2)  # (B*T,2)
    x_abs_flat_norm = x_abs_seq_bt.cpu().numpy().reshape(-1)  # normalized |x|
    x_theta_flat = x_theta_seq_bt.cpu().numpy().reshape(-1)
    targets_flat_norm = targets_seq_t.cpu().numpy().reshape(-1, 2)

# -----------------------------
# De-normalize to original scale
# -----------------------------
# Model outputs are normalized by y_rms; bring them back:
outputs_denorm = outputs_np * y_rms                    # (Ncut,2) in original scale
targets_denorm = targets_flat_norm * y_rms             # (Ncut,2) in original scale
x_abs_denorm = x_abs_flat_norm * x_rms                 # (Ncut,) original |x|

# Complex forms
y_pred_complex = to_complex(outputs_denorm)            # (Ncut,)
y_real_complex = to_complex(targets_denorm)            # (Ncut,)

# Save predicted complex for any downstream diagnostics
savemat("outputs_complex.mat", {"outputs_complex": y_pred_complex})

# -----------------------------
# AM/AM (original scale)
# -----------------------------
y_pred_mag = np.abs(y_pred_complex)
y_real_mag = np.abs(y_real_complex)

plt.figure(figsize=(8, 6))
plt.scatter(x_abs_denorm, y_real_mag, s=1, label='Target (Original distorted)')
plt.scatter(x_abs_denorm, y_pred_mag, s=1, label='Inverse PG-JANET Output (de-norm)')
plt.xlabel("Input Magnitude |X| (original)")
plt.ylabel("Output Magnitude |Y| (original)")
plt.title("AM/AM: Target vs Inverse PG-JANET (de-normalized)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("inverse_pg_janet_am_am.png", dpi=140)
plt.show()

# -----------------------------
# AM/PM (original scale)
# -----------------------------
B = x_abs_seq_bt.shape[0]

with torch.no_grad():
    # Recreate sequence-wise complex on original scale for phase
    outputs_btc = outputs_bt.cpu().numpy() * y_rms  # (B,T,2) de-norm
    targets_btc = targets_seq_np * y_rms            # (B,T,2) de-norm (already on CPU)

x_mag_all_list, ampm_pred_list, ampm_real_list = [], [], []
for i in range(B):
    x_mag_seq = (x_abs_seq_np[i].flatten()) * x_rms
    x_theta_seq_local = x_theta_seq_np[i].flatten()      # <-- fixed: no shadowing
    y_pred_seq = outputs_btc[i]
    y_real_seq = targets_btc[i]

    y_pred_c = y_pred_seq[:, 0] + 1j * y_pred_seq[:, 1]
    y_real_c = y_real_seq[:, 0] + 1j * y_real_seq[:, 1]

    x_theta_unwrap = np.unwrap(x_theta_seq_local)
    y_pred_phase = np.unwrap(np.angle(y_pred_c))
    y_real_phase = np.unwrap(np.angle(y_real_c))

    ampm_pred = y_pred_phase - x_theta_unwrap
    ampm_real = y_real_phase - x_theta_unwrap

    x_mag_all_list.append(x_mag_seq)
    ampm_pred_list.append(ampm_pred)
    ampm_real_list.append(ampm_real)

x_mag_all = np.concatenate(x_mag_all_list)
am_pm_pred = np.concatenate(ampm_pred_list)
am_pm_real = np.concatenate(ampm_real_list)

# Center both around the same mean for visual comparison
center = np.mean(am_pm_real)
am_pm_real -= center
am_pm_pred -= center

plt.figure(figsize=(8, 6))
plt.scatter(x_mag_all, am_pm_real, s=1, label='Target (Original distorted)')
plt.scatter(x_mag_all, am_pm_pred, s=1, label='Inverse PG-JANET Output (de-norm)')
plt.xlabel("Input Magnitude |X| (original)")
plt.ylabel("Phase Shift (radians)")
plt.title("AM/PM: Target vs Inverse PG-JANET (de-normalized)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("inverse_pg_janet_am_pm.png", dpi=140)
plt.show()

# -----------------------------
# Spectrum (original scale)
# -----------------------------
# Rebuild full complex waveforms on original scale for spectrum
x_complex = x_abs_denorm * np.exp(1j * x_theta_flat)  # original input (clean)

# Compose predicted/target complex from de-norm AM/PM relative to x phase:
# We already have de-norm amplitudes from y_pred_mag / y_real_mag, and AM/PM from above.
# am_pm_pred/real are aligned with concatenated sequences and with x_theta_flat.
y_pred_sig = from_mag_phase(y_pred_mag, am_pm_pred + x_theta_flat)
y_real_sig = from_mag_phase(y_real_mag, am_pm_real + x_theta_flat)

def plot_spectrum(sig: np.ndarray, label: str , color: str ):
    spectrum = fftshift(fft(sig))
    spectrum_db = 20 * np.log10(np.abs(spectrum) + 1e-12)-46.3  # dBm conversion
    spectrum_db_smooth = sg_smooth(spectrum_db, 101, 3)
    N = len(sig)
    f = np.linspace(-0.5, 0.5, N)
    plt.plot(f, spectrum_db_smooth, label=label, linewidth=2, color=color)

plt.figure(figsize=(10, 6))
plot_spectrum(x_complex, 'Original Y (distorted)', 'blue')
plot_spectrum(y_real_sig, 'Original Target (clean)', 'black')
plot_spectrum(y_pred_sig, 'Inverse PG-JANET Output (de-norm)', 'red')
plt.title("Spectrum Comparison (FFT, smoothed) — Original Scale")
plt.xlabel("Normalized Frequency")
plt.ylabel("Magnitude (dB)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("inverse_pg_janet_spectrum.png", dpi=140)
plt.show()
