import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from scipy.io import loadmat
from pg_janet import PGJanetRNN  # ודא שקיים במערכת שלך
import matplotlib.pyplot as plt

# =========================
# Loss
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
    num = np.vdot(x, y)  # conj(x)·y
    den = np.vdot(x, x) + 1e-12
    return num / den

def complex_to_2chan(z: np.ndarray) -> np.ndarray:
    return np.stack([np.real(z), np.imag(z)], axis=-1).astype(np.float32)

# =========================
# Dataset (GLOBAL RMS)
# =========================
class DirectDPDDataset(Dataset):
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
    def __init__(self, hidden_size=64):
        super().__init__()
        self.core = PGJanetRNN(hidden_size=hidden_size)

    def forward(self, x_abs_norm: torch.Tensor, x_theta: torch.Tensor) -> torch.Tensor:
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
                 fwd_ckpt='pg_janet_forward_global_rms_new.pth',
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

        # Frozen PA forward model (must match how you trained it)
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

        self.sched = optim.lr_scheduler.ReduceLROnPlateau(
            self.optim, mode='min', patience=1, factor=0.7, threshold=1e-3, min_lr=1e-7
        )

        self.n_epochs = n_epochs
        self.seq_len = seq_len
        self.history = {"train": [], "val": [], "lrs": []}

    @staticmethod
    def _complex_from_reim(t: torch.Tensor) -> torch.Tensor:
        return torch.view_as_complex(t)

    def _pa_forward(self, u_reim: torch.Tensor) -> torch.Tensor:
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
            x_th  = x_th.to(self.device)               # (B,T)
            y_lin = y_lin.to(self.device)               # (B,T,2)

            u_pred = self.pred(x_abs, x_th)              # (B,T,2)
            y_hat  = self._pa_forward(u_pred)            # (B,T,2)

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
        n_epochs=100,
        lr=1e-3,
        fwd_ckpt="pg_janet_forward_global_rms_new.pth"
    )
    trainer.train_loop()
    trainer.save("pg_janet_predistorter_direct.pth")
