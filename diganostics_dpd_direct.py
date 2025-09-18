import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from scipy.fft import fft, fftshift
from scipy.signal import savgol_filter as sg_smooth
from pg_janet import PGJanetRNN  # ודא שקיים


class PredistorterRNN(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.core = PGJanetRNN(hidden_size=hidden_size)

    def forward(self, x_abs_norm: torch.Tensor, x_theta: torch.Tensor) -> torch.Tensor:
        return self.core(x_abs_norm, x_theta)


def compute_global_rms(arr, epsilon=1e-8) -> float:
    return float(np.sqrt(np.mean(np.abs(arr) ** 2) + epsilon))

def ls_linear_gain(x: np.ndarray, y: np.ndarray) -> complex:
    num = np.vdot(x, y)  # conj(x)·y
    den = np.vdot(x, x) + 1e-12
    return num / den

def complex_to_2chan(z: np.ndarray) -> np.ndarray:
    return np.stack([np.real(z), np.imag(z)], axis=-1).astype(np.float32)

def from_mag_phase(mag: np.ndarray, phase: np.ndarray) -> np.ndarray:
    return mag * np.exp(1j * phase)

def plot_spectrum(sig: np.ndarray, label: str, linewidth=2, color=None):
    spectrum = fftshift(fft(sig))
    spectrum_db = 20 * np.log10(np.abs(spectrum) + 1e-12) - 46.3  # dBm conversion (as in המקור)
    spectrum_db_smooth = sg_smooth(spectrum_db, 101, 3)
    N = len(sig)
    f = np.linspace(-0.5, 0.5, N)
    if color:
        plt.plot(f, spectrum_db_smooth, label=label, linewidth=linewidth, color=color)
    else:
        plt.plot(f, spectrum_db_smooth, label=label, linewidth=linewidth)

def diagnostics(mat_path='DPD_signal.mat',
                fwd_ckpt='pg_janet_forward_global_rms_new.pth',
                pred_ckpt='pg_janet_predistorter_direct.pth',
                pd_hidden=32,
                fwd_hidden=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load mat
    mat = loadmat(mat_path, squeeze_me=True)
    X = mat['sig_in']   # complex clean input
    Y = mat['sig_out']  # complex distorted output

    x_rms = compute_global_rms(np.abs(X))
    y_rms = compute_global_rms(np.abs(Y))
    x_abs = (np.abs(X).astype(np.float32) / (x_rms + 1e-8)).reshape(-1, 1)
    x_abs = x_abs * 10**(-1/20)  # כפי שהיה במקור
    x_theta = np.angle(X).astype(np.float32)
    g_lin = ls_linear_gain(X, Y)
    Y_lin = g_lin * X
    Y_lin_norm = Y_lin / (y_rms + 1e-8)
    targets = complex_to_2chan(Y_lin_norm)  # (N,2)

    N = len(x_abs)
    # Flatten sequences for diagnostics (כל דגימה)
    x_abs_seq = np.array([x_abs[i] for i in range(N)])
    x_theta_seq = np.array([x_theta[i] for i in range(N)])
    targets_seq = np.array(targets)

    # Tensors
    x_abs_torch = torch.from_numpy(x_abs_seq).float().unsqueeze(0).to(device)  # (1,N,1)
    x_theta_torch = torch.from_numpy(x_theta_seq).float().unsqueeze(0).to(device)  # (1,N)
    targets_torch = torch.from_numpy(targets_seq).float().unsqueeze(0).to(device)  # (1,N,2)

    # Load trained predistorter
    pred = PredistorterRNN(hidden_size=pd_hidden).to(device)
    pred.load_state_dict(torch.load(pred_ckpt, map_location=device))
    pred.eval()

    # Load forward PA (for pa forward function)
    pa = PGJanetRNN(hidden_size=fwd_hidden).to(device)
    pa.load_state_dict(torch.load(fwd_ckpt, map_location=device))
    pa.eval()

    def pa_forward(u_reim: torch.Tensor) -> torch.Tensor:
        u_re = u_reim[..., 0]
        u_im = u_reim[..., 1]
        u_abs = torch.sqrt(u_re ** 2 + u_im ** 2 + 1e-12)
        u_th  = torch.atan2(u_im, u_re)
        return pa(u_abs, u_th)

    with torch.no_grad():
        # note: original code used pred(x_abs_torch.squeeze(-1), x_theta_torch)
        u_pred = pred(x_abs_torch.squeeze(-1), x_theta_torch)  # (1,N,2)
        y_pred = pa_forward(u_pred)                            # (1,N,2)

    u_pred_np = u_pred.cpu().numpy().reshape(-1, 2)
    y_pred_np = y_pred.cpu().numpy().reshape(-1, 2)
    x_abs_flat = x_abs_torch.cpu().numpy().reshape(-1)
    x_theta_flat = x_theta_torch.cpu().numpy().reshape(-1)
    targets_flat = targets_torch.cpu().numpy().reshape(-1, 2)

    # AM/AM and AM/PM
    y_pred_mag = np.sqrt(y_pred_np[:, 0]**2 + y_pred_np[:, 1]**2)
    y_pred_phase = np.arctan2(y_pred_np[:, 1], y_pred_np[:, 0])
    y_real_mag = np.sqrt(targets_flat[:, 0]**2 + targets_flat[:, 1]**2)
    y_real_phase = np.arctan2(targets_flat[:, 1], targets_flat[:, 0])
    am_pm_pred = y_pred_phase - x_theta_flat
    am_pm_real = y_real_phase - x_theta_flat

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(x_abs_flat * x_rms, y_pred_mag * y_rms, s=1, alpha=0.3, label='Predicted')
    plt.scatter(x_abs_flat * x_rms, y_real_mag * y_rms, s=1, alpha=0.3, label='Target')
    plt.xlabel('|x| (Input Amplitude)'); plt.ylabel('|y| (Output Amplitude)')
    plt.title('AM/AM Characteristic (Original Scale)'); plt.legend(); plt.grid()
    plt.subplot(1, 2, 2)
    plt.scatter(x_abs_flat * x_rms, am_pm_pred * 180/np.pi, s=1, alpha=0.3, label='Predicted')
    plt.scatter(x_abs_flat * x_rms, am_pm_real * 180/np.pi, s=1, alpha=0.3, label='Target')
    plt.xlabel('|x| (Input Amplitude)'); plt.ylabel('Phase Difference (Degrees)')
    plt.title('AM/PM Characteristic (Original Scale)'); plt.legend(); plt.grid()
    plt.tight_layout()
    plt.savefig('direct_dpd_am_am_pm.png', dpi=140)
    plt.show()

    # Spectrum plot (de-normalize back to original scale)
    x_abs_denorm = x_abs_flat * x_rms
    y_pred_denorm = from_mag_phase(y_pred_mag * y_rms, am_pm_pred + x_theta_flat)
    y_real_denorm = from_mag_phase(y_real_mag * y_rms, am_pm_real + x_theta_flat)
    x_complex = from_mag_phase(x_abs_denorm, x_theta_flat)
    y_pred_sig = y_pred_denorm
    y_real_sig = y_real_denorm

    plt.figure(figsize=(10, 6))
    plot_spectrum(x_complex, 'Original Input (clean)', color='black')
    plot_spectrum(y_real_sig, 'Original Target (distorted)', color='blue')
    plot_spectrum(y_pred_sig, 'Predistorted Output (de-norm)', color='red')
    plt.title("Spectrum Comparison (FFT, smoothed) — Original Scale")
    plt.xlabel("Normalized Frequency")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("direct_dpd_spectrum.png", dpi=140)
    plt.show()
    plt.close()

    # Save sum of signals as .mat
    def save_sum_of_signals(x_complex: np.ndarray, u_pred_np: np.ndarray, save_path: str = "sum_signals.mat"):
        u_complex = u_pred_np[:, 0] + 1j * u_pred_np[:, 1]
        min_len = min(len(x_complex), len(u_complex))
        x_c = x_complex[:min_len]
        u_c = u_complex[:min_len]
        sum_signal = x_c + u_c
        savemat(save_path, {"x_complex": x_c, "u_complex": u_c, "sum_signal": sum_signal})
        print(f"Saved x_complex, u_complex, and their sum to {save_path}")

    save_sum_of_signals(x_complex, u_pred_np, "sum_signals.mat")

if __name__ == "__main__":
    diagnostics(
        mat_path='DPD_signal.mat',
        fwd_ckpt='pg_janet_forward_global_rms_new.pth',
        pred_ckpt='pg_janet_predistorter_direct.pth',
        pd_hidden=32,
        fwd_hidden=32
    )
