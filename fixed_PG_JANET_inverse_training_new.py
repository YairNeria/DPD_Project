import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from scipy.io import loadmat
from pg_janet import PGJanetRNN
import matplotlib.pyplot as plt

# -----------------------------
# Loss
# -----------------------------
class nMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, y_pred, y_true):
        mse = torch.sum((y_pred - y_true) ** 2)
        denom = torch.sum(y_true ** 2) + 1e-8
        return mse / denom

# -----------------------------
# Global RMS helpers
# -----------------------------
def compute_global_rms(arr: np.ndarray, epsilon: float = 1e-8) -> float:
    """Compute a single RMS over the entire 1D/2D array."""
    return float(np.sqrt(np.mean(arr**2) + epsilon))

# (optional) keep these around if you still want to experiment elsewhere
def max_norm(x, max_val=None):
    if max_val is None:
        max_val = np.max(np.abs(x))
    return x 

def std_norm(x, mean=None, std=None):
    if mean is None:
        mean = x.mean()
    if std is None:
        std = x.std()
    return (x - mean) / (std + 1e-8)

def average_norm(x):
    mean = x.mean()
    std = x.std()
    return (x - mean) / (std + 1e-8)

# -----------------------------
# Dataset with GLOBAL RMS normalization
# -----------------------------
class PGJanetSequenceDataset(Dataset):
    def __init__(self, mat_path='for_DPD.mat', seq_len=10, invert=True):
        mat = loadmat(mat_path, squeeze_me=True)

        # Choose direction
        if not invert:
            X = mat['sig_in']
            Y = mat['sig_out']
        else:
            X = mat['sig_out']
            Y = mat['sig_in']

        # Base arrays
        x_abs = np.abs(X).astype(np.float32).reshape(-1, 1)  # (N, 1)
        x_theta = np.angle(X).astype(np.float32)             # (N,)
        y_abs = np.abs(Y).astype(np.float32)                 # (N,)
        y_theta = np.angle(Y).astype(np.float32)             # (N,)

        # -------- GLOBAL RMS normalization (one scale for whole dataset) --------
        x_rms = compute_global_rms(x_abs)        # scalar
        y_rms = compute_global_rms(y_abs)        # scalar

        x_abs_norm = x_abs / (x_rms + 1e-8)      # (N, 1)
        y_abs_norm = y_abs / (y_rms + 1e-8)      # (N,)
        y_abs_norm = np.clip(y_abs_norm, 0, None)

        # Build complex targets (normalized)
        Y_norm = y_abs_norm * np.exp(1j * y_theta)
        targets = np.stack([np.real(Y_norm), np.imag(Y_norm)], axis=-1).astype(np.float32)  # (N, 2)

        # Store factors for de-normalization at inference time
        self.x_rms = x_rms
        self.y_rms = y_rms

        # Slice into sequences
        self.x_abs_seq = []
        self.theta_seq = []
        self.target_seq = []
        N = len(x_abs_norm)
        for i in range(N - seq_len):
            self.x_abs_seq.append(x_abs_norm[i:i+seq_len])      # (seq_len, 1)
            self.theta_seq.append(x_theta[i:i+seq_len])         # (seq_len,)
            self.target_seq.append(targets[i:i+seq_len])        # (seq_len, 2)

        self.x_abs_seq = torch.from_numpy(np.array(self.x_abs_seq))   # (num_seq, seq_len, 1)
        self.theta_seq = torch.from_numpy(np.array(self.theta_seq))    # (num_seq, seq_len)
        self.target_seq = torch.from_numpy(np.array(self.target_seq))  # (num_seq, seq_len, 2)

    def __len__(self):
        return self.x_abs_seq.shape[0]

    def __getitem__(self, idx):
        return (
            self.x_abs_seq[idx],
            self.theta_seq[idx],
            self.target_seq[idx]
        )

# -----------------------------
# Trainer
# -----------------------------
class TrainModel(nn.Module):
    def __init__(self, seq_len=10, hidden_size=64, n_epochs=30, batch_size=64,
                 mat_path='for_DPD.mat', learning_rate=1e-2, invert=True):
        super(TrainModel, self).__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.mat_path = mat_path
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print('Loading dataset...')
        self.dataset = PGJanetSequenceDataset(mat_path, seq_len=seq_len, invert=invert)
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        self.model = PGJanetRNN(hidden_size=hidden_size).to(self.device)
        self.loss_function = nMSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=1, factor=0.7,
            threshold=1e-3, threshold_mode='rel', cooldown=0, min_lr=1e-7
        )

        self.learning_rates = []
        self.epoch_losses_list = []
        self.val_losses_list = []

        # train/val split
        val_ratio = 0.25
        n_val = int(len(self.dataset) * val_ratio)
        n_train = len(self.dataset) - n_val
        self.train_set, self.val_set = random_split(self.dataset, [n_train, n_val])
        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_set, batch_size=batch_size, shuffle=False)

    def train(self):
        print('Starting training...')
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            self.model.train()
            for batch_x_abs, batch_theta, batch_targets in self.train_loader:
                batch_x_abs = batch_x_abs.to(self.device).squeeze(-1)  # (B, T)
                batch_theta = batch_theta.to(self.device).squeeze(-1)  # (B, T)
                batch_targets = batch_targets.to(self.device)          # (B, T, 2)

                outputs = self.model(batch_x_abs, batch_theta)         # (B, T, 2)
                loss = self.loss_function(outputs, batch_targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= len(self.train_loader)
            self.epoch_losses_list.append(epoch_loss)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for val_x_abs, val_theta, val_targets in self.val_loader:
                    val_x_abs = val_x_abs.to(self.device).squeeze(-1)
                    val_theta = val_theta.to(self.device).squeeze(-1)
                    val_targets = val_targets.to(self.device)
                    val_outputs = self.model(val_x_abs, val_theta)
                    val_loss += self.loss_function(val_outputs, val_targets).item()
                val_loss /= len(self.val_loader)
                self.val_losses_list.append(val_loss)

            self.scheduler.step(val_loss)
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
            print(f"Epoch {epoch+1} | train: {epoch_loss:.6g} | val: {val_loss:.6g} | lr: {self.optimizer.param_groups[0]['lr']:.3e}")



        print('Training complete.')
        # Plot in dB
        loss_db = 10 * np.log10(np.array(self.epoch_losses_list) + 1e-12)
        val_db  = 10 * np.log10(np.array(self.val_losses_list) + 1e-12)
        fig, ax1 = plt.subplots()
        ax1.plot(loss_db, 'b-', label='Train Loss (dB)')
        ax1.plot(val_db, 'g-', label='Val Loss (dB)')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (dB)', color='b')
        ax2 = ax1.twinx()
        ax2.plot(self.learning_rates, 'r--', label='Learning Rate')
        ax2.set_ylabel('Learning Rate', color='r')
        fig.legend(loc='upper right')
        plt.title('Training and Validation Loss (dB) and Learning Rate per Epoch')
        fig.tight_layout()
        plt.show()
        fig.savefig('inverse_pg_janet_loss.png')

    def save_model(self, path='pg_janet_inverse.pth'):
        print(f'Saving model to {path}...')
        torch.save(self.model.state_dict(), path)
        print('Model saved.')

if __name__ == "__main__":
    seq_len = 16
    hidden_size = 32
    train_model = TrainModel(seq_len=seq_len,
                             hidden_size=hidden_size,
                             mat_path='DPD_signal (1).mat',
                             invert=False)
    train_model.train()
    train_model.save_model('pg_janet_forward_global_rms_new.pth')
# -----------------------------