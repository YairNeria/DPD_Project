import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from scipy.io import loadmat
from pg_janet import PGJanetRNN  # same model family you already use
import matplotlib.pyplot as plt

# =========================
# Loss (same as your code)
# =========================
class nMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, y_pred, y_true):
        mse = torch.sum((y_pred - y_true) ** 2)
        denom = torch.sum(y_true ** 2) + 1e-8
        return mse / denom

# =========================
# Helpers
# =========================
def compute_global_rms(arr, epsilon=1e-8) -> float:
    return float(np.sqrt(np.mean(np.abs(arr) ** 2) + epsilon))

def ls_linear_gain(x: np.ndarray, y: np.ndarray) -> complex:
    # g = argmin || y - g x ||^2  => g = (x^H y)/(x^H x)
    num = np.vdot(x, y)  # conj(x)·y
    den = np.vdot(x, x) + 1e-12
    return num / den

def complex_to_2chan(z: np.ndarray) -> np.ndarray:
    return np.stack([np.real(z), np.imag(z)], axis=-1).astype(np.float32)

# =========================
# Dataset (GLOBAL RMS)
# =========================
class DirectDPDDataset(Dataset):
    """
    Input:  sig_in  (clean)  -> x_abs_norm, x_theta
    Target: y_lin = g * sig_in, normalized by y_rms, as (re, im)

    We'll train predistorter P so that  F( P(x) ) ≈ y_lin,
    where F is a frozen forward PA model.
    """
    def __init__(self, mat_path='DPD_signal.mat', seq_len=16):
        mat = loadmat(mat_path, squeeze_me=True)
        X = mat['sig_in']   # complex clean input
        Y = mat['sig_out']  # complex distorted output

        # Global scaling
        self.x_rms = compute_global_rms(np.abs(X))
        self.y_rms = compute_global_rms(np.abs(Y))

        x_abs = (np.abs(X).astype(np.float32) / (self.x_rms + 1e-8)).reshape(-1, 1)
        x_theta = np.angle(X).astype(np.float32)

        # Linear target y_lin = g * X (choose LS gain)
        g_lin = ls_linear_gain(X, Y)
        self.g_lin = g_lin
        Y_lin = g_lin * X
        Y_lin_norm = Y_lin / (self.y_rms + 1e-8)
        targets = complex_to_2chan(Y_lin_norm)  # (N,2)

        # Slice into sequences
        self.seq_len = seq_len
        self.x_abs_seq, self.theta_seq, self.target_seq = [], [], []
        N = len(x_abs)
        for i in range(N - seq_len):
            self.x_abs_seq.append(x_abs[i:i+seq_len])         # (T,1)
            self.theta_seq.append(x_theta[i:i+seq_len])       # (T,)
            self.target_seq.append(targets[i:i+seq_len])      # (T,2)

        self.x_abs_seq = torch.from_numpy(np.array(self.x_abs_seq)).float()
        self.theta_seq = torch.from_numpy(np.array(self.theta_seq)).float()
        self.target_seq = torch.from_numpy(np.array(self.target_seq)).float()

    def __len__(self):
        return self.x_abs_seq.shape[0]

    def __getitem__(self, idx):
        return self.x_abs_seq[idx], self.theta_seq[idx], self.target_seq[idx]

# =========================
# Predistorter Wrapper
# =========================
class PredistorterRNN(nn.Module):
    """
    Wrap PGJanetRNN to output complex predistorted signal (re, im) normalized by x_rms.
    We'll convert (re,im) -> (abs,theta) before feeding to the PA forward model.
    """
    def __init__(self, hidden_size=64):
        super().__init__()
        self.core = PGJanetRNN(hidden_size=hidden_size)

    def forward(self, x_abs_norm: torch.Tensor, x_theta: torch.Tensor) -> torch.Tensor:
        """
        x_abs_norm: (B,T)  normalized by x_rms
        x_theta:    (B,T)
        returns:    (B,T,2)  predistorted complex (re, im) at input normalization scale (x_rms)
        """
        # Reuse the same interface as your model: (B,T)->(B,T,2)
        # You can also make a separate head if you prefer.
        return self.core(x_abs_norm, x_theta)

# =========================
# Training Loop
# =========================
class TrainDirectDPD(nn.Module):
    def __init__(self,
                 mat_path='DPD_signal.mat',
                 seq_len=16,
                 pd_hidden=64,
                 fwd_hidden=64,
                 batch_size=64,
                 n_epochs=25,
                 lr=1e-3,
                 fwd_ckpt='pg_janet_forward_global_rms.pth',
                 device=None):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("Loading dataset...")
        self.dataset = DirectDPDDataset(mat_path=mat_path, seq_len=seq_len)
        val_ratio = 0.25
        n_val = int(len(self.dataset) * val_ratio)
        n_train = len(self.dataset) - n_val
        self.train_set, self.val_set = random_split(self.dataset, [n_train, n_val])
        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_set, batch_size=batch_size, shuffle=False)

        # Predistorter to train
        self.pred = PredistorterRNN(hidden_size=pd_hidden).to(self.device)

        # Frozen PA forward model (must match how you trained it: x_abs/x_theta -> y(re,im))
        self.pa = PGJanetRNN(hidden_size=fwd_hidden).to(self.device)
        if not os.path.exists(fwd_ckpt):
            raise FileNotFoundError(
                f"Forward PA checkpoint '{fwd_ckpt}' not found. "
                "Train your PA model (sig_in->sig_out) with the same normalization first."
            )
        self.pa.load_state_dict(torch.load(fwd_ckpt, map_location=self.device))
        self.pa.eval()
        for p in self.pa.parameters():
            p.requires_grad_(False)

        self.loss_fn = nMSELoss()
        self.optim = optim.Adam(self.pred.parameters(), lr=lr)

        # Plateau scheduler on **validation** loss (not train loss!)
        self.sched = optim.lr_scheduler.ReduceLROnPlateau(
            self.optim, mode='min', patience=1, factor=0.7, threshold=1e-3, min_lr=1e-7
        )

        self.n_epochs = n_epochs
        self.seq_len = seq_len
        self.history = {"train": [], "val": [], "lrs": []}

    @staticmethod
    def _complex_from_reim(t: torch.Tensor) -> torch.Tensor:
        # (B,T,2) -> complex (B,T) in PyTorch's complex dtype
        return torch.view_as_complex(t)

    def _pa_forward(self, u_reim: torch.Tensor) -> torch.Tensor:
        """
        u_reim: (B,T,2) predistorted complex @ x_rms scale (normalized)
        Convert to (abs, theta) then run frozen PA model -> (B,T,2) in y_rms scale (normalized)
        """
        u_re = u_reim[..., 0]
        u_im = u_reim[..., 1]
        u_abs = torch.sqrt(u_re ** 2 + u_im ** 2 + 1e-12)
        u_th  = torch.atan2(u_im, u_re)
        return self.pa(u_abs, u_th)

    def run_epoch(self, train=True):
        loader = self.train_loader if train else self.val_loader
        self.pred.train(train)

        total = 0.0
        for x_abs, x_th, y_lin in loader:
            x_abs = x_abs.to(self.device).squeeze(-1)   # (B,T)
            x_th  = x_th.to(self.device)                 # (B,T)
            y_lin = y_lin.to(self.device)                # (B,T,2)

            # Forward: predistorter -> PA -> predicted output
            u_pred = self.pred(x_abs, x_th)              # (B,T,2) normalized @ x_rms
            y_hat  = self._pa_forward(u_pred)            # (B,T,2) normalized @ y_rms

            loss = self.loss_fn(y_hat, y_lin)

            if train:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            total += loss.item()
        return total / len(loader)

    def train_loop(self):
        print("Starting direct DPD training...")
        for ep in range(1, self.n_epochs + 1):
            tr = self.run_epoch(train=True)
            vl = self.run_epoch(train=False)
            self.sched.step(vl)
            self.history["train"].append(tr)
            self.history["val"].append(vl)
            self.history["lrs"].append(self.optim.param_groups[0]["lr"])

            print(f"Epoch {ep:02d} | train={tr:.6g} | val={vl:.6g} | lr={self.optim.param_groups[0]['lr']:.3e}")

        print("Training complete.")
        # Quick plot
        loss_db = 10*np.log10(np.array(self.history["train"]) + 1e-12)
        val_db  = 10*np.log10(np.array(self.history["val"]) + 1e-12)
        fig, ax1 = plt.subplots()
        ax1.plot(loss_db, label='Train Loss (dB)')
        ax1.plot(val_db,  label='Val Loss (dB)')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss (dB)')
        ax2 = ax1.twinx(); ax2.plot(self.history["lrs"], 'r--', label='LR')
        ax2.set_ylabel('Learning Rate')
        fig.legend(loc='upper right'); fig.tight_layout(); plt.show()
        fig.savefig('direct_dpd_loss.png', dpi=140)

    def save(self, path='pg_janet_predistorter_direct.pth'):
        torch.save(self.pred.state_dict(), path)
        print(f"Saved predistorter to {path}")

if __name__ == "__main__":
    trainer = TrainDirectDPD(
        mat_path="DPD_signal.mat",
        seq_len=16,
        pd_hidden=32,
        fwd_hidden=32,
        batch_size=64,
        n_epochs=25,
        lr=1e-3,
        fwd_ckpt="pg_janet_forward_global_rms.pth"  # <-- set to your forward PA model weights
    )
    trainer.train_loop()
    trainer.save("pg_janet_predistorter_direct.pth")

# =========================
# Diagnostics (AM/AM, AM/PM, Spectrum)
# =========================
# Load the entire dataset for diagnostics
mat = loadmat('DPD_signal.mat', squeeze_me=True)
X = mat['sig_in']   # complex clean input
Y = mat['sig_out']  # complex distorted output
x_rms = compute_global_rms(np.abs(X))
y_rms = compute_global_rms(np.abs(Y))
x_abs = (np.abs(X).astype(np.float32) / (x_rms + 1e-8)).reshape(-1, 1)
x_theta = np.angle(X).astype(np.float32)
g_lin = ls_linear_gain(X, Y)
Y_lin = g_lin * X
Y_lin_norm = Y_lin / (y_rms + 1e-8)
targets = complex_to_2chan(Y_lin_norm)  # (N,2)
N = len(x_abs)
x_abs_seq = []
x_theta_seq = []
for i in range(N):
    x_abs_seq.append(x_abs[i])
    x_theta_seq.append(x_theta[i])
x_abs_seq = np.array(x_abs_seq)
x_theta_seq = np.array(x_theta_seq)
targets_seq = np.array(targets)
# Convert to tensors
x_abs_torch = torch.from_numpy(x_abs_seq).float().unsqueeze(0).to(trainer.device)  # (1,N,1)
x_theta_torch = torch.from_numpy(x_theta_seq).float().unsqueeze(0).to(trainer.device)  # (1,N)
targets_torch = torch.from_numpy(targets_seq).float().unsqueeze(0).to(trainer.device)  # (1,N,2)     
# Load trained predistorter
pred = PredistorterRNN(hidden_size=32).to(trainer.device)
pred.load_state_dict(torch.load("pg_janet_predistorter_direct.pth", map_location=trainer.device))
pred.eval()
with torch.no_grad():
    u_pred = pred(x_abs_torch.squeeze(-1), x_theta_torch)  # (1,N,2)
    y_pred = trainer._pa_forward(u_pred)                    # (1,N,2)   
u_pred_np = u_pred.cpu().numpy().reshape(-1, 2)
y_pred_np = y_pred.cpu().numpy().reshape(-1, 2)
x_abs_flat = x_abs_torch.cpu().numpy().reshape(-1)
x_theta_flat = x_theta_torch.cpu().numpy().reshape(-1)
targets_flat = targets_torch.cpu().numpy().reshape(-1, 2)
# AM/AM and AM/PM plots
def from_mag_phase(mag: np.ndarray, phase: np.ndarray) -> np.ndarray:
    return mag * np.exp(1j * phase)
y_pred_mag = np.sqrt(y_pred_np[:, 0]**2 + y_pred_np[:, 1]**2)
y_pred_phase = np.arctan2(y_pred_np[:, 1], y_pred_np[:, 0])
y_real_mag = np.sqrt(targets_flat[:, 0]**2 + targets_flat[:, 1]**2)
y_real_phase = np.arctan2(targets_flat[:, 1], targets_flat[:, 0])
am_pm_pred = y_pred_phase - x_theta_flat
am_pm_real = y_real_phase - x_theta_flat
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(x_abs_flat * x_rms, y_pred_mag * y_rms, s=1, alpha=0.3, label='Predicted', color='C1')
plt.scatter(x_abs_flat * x_rms, y_real_mag * y_rms, s=1, alpha=0.3, label='Target', color='C0')
plt.xlabel('|x| (Input Amplitude)'); plt.ylabel('|y| (Output Amplitude)')
plt.title('AM/AM Characteristic (Original Scale)'); plt.legend(); plt.grid()
plt.subplot(1, 2, 2)
plt.scatter(x_abs_flat * x_rms, am_pm_pred * 180/np.pi, s=1, alpha=0.3, label='Predicted', color='C1')
plt.scatter(x_abs_flat * x_rms, am_pm_real * 180/np.pi, s=1, alpha=0.3, label='Target', color='C0')
plt.xlabel('|x| (Input Amplitude)'); plt.ylabel('Phase Difference (Degrees)')
plt.title('AM/PM Characteristic (Original Scale)'); plt.legend(); plt.grid()
plt.tight_layout(); plt.show()
plt.savefig('direct_dpd_am_am_pm.png', dpi=140)
# Spectrum plot
from scipy.fft import fft, fftshift
from scipy.signal import savgol_filter as sg_smooth
x_abs_denorm = x_abs_flat * x_rms
y_pred_denorm = from_mag_phase(y_pred_mag * y_rms, am_pm_pred + x_theta_flat)
y_real_denorm = from_mag_phase(y_real_mag * y_rms, am_pm_real + x_theta_flat)
x_complex = from_mag_phase(x_abs_denorm, x_theta_flat)
y_pred_sig = y_pred_denorm
y_real_sig = y_real_denorm
def plot_spectrum(sig: np.ndarray, label: str , color: str = 'C0'):
    spectrum = fftshift(fft(sig))
    spectrum_db = 20 * np.log10(np.abs(spectrum) + 1e-12)-46.3  # dBm conversion
    spectrum_db_smooth = sg_smooth(spectrum_db, 101, 3)
    N = len(sig)
    f = np.linspace(-0.5, 0.5, N)
    plt.plot(f, spectrum_db_smooth, label=label, linewidth=2, color=color)
plt.figure(figsize=(10, 6))
plot_spectrum(x_complex, 'Original Input (clean)', 'black')
plot_spectrum(y_real_sig, 'Original Target (distorted)', 'blue')
plot_spectrum(y_pred_sig, 'Predistorted Output (de-norm)', 'red')
plt.title("Spectrum Comparison (FFT, smoothed) — Original Scale")
plt.xlabel("Normalized Frequency")
plt.ylabel("Magnitude (dB)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("direct_dpd_spectrum.png", dpi=140)
plt.show()
plt.close()
# -----------------------------
# The End