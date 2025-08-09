import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from scipy.io import loadmat
from pg_janet import PGJanetRNN
import matplotlib.pyplot as plt

# Custom nMSE loss function (normalized mean square error)
class nMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, y_pred, y_true):
        mse = torch.sum((y_pred - y_true) ** 2)
        denom = torch.sum(y_true ** 2) + 1e-8
        return mse / denom

# Normalize by max value (scales to [0, 1])
def max_norm(x, max_val=None):
    if max_val is None:
        max_val = np.max(np.abs(x))
    return x / (max_val + 1e-8)

# Standard score normalization (mean=0, std=1)
def std_norm(x, mean=None, std=None):
    if mean is None:
        mean = x.mean()
    if std is None:
        std = x.std()
    return (x - mean) / (std + 1e-8)

class PGJanetSequenceDataset(Dataset):
    def __init__(self, mat_path='U_and_U_ideal_for_pjanet.mat', seq_len=10, invert=False, norm_func=max_norm):
        # Load .mat file
        mat = loadmat(mat_path, squeeze_me=True)
        # Choose signal direction (normal or inverted)
        if not invert:
            X = mat['u']
            Y = mat['u_ideal']
        else:
            X = mat['u_ideal']
            Y = mat['u']

        x_abs = np.abs(X).astype(np.float32).reshape(-1, 1)
        x_theta = np.angle(X).astype(np.float32)
        x_abs = norm_func(x_abs)

        y_abs = np.abs(Y).astype(np.float32)
        y_theta = np.angle(Y).astype(np.float32)
        y_abs_max = np.max(y_abs)
        y_abs_norm = max_norm(y_abs, y_abs_max)
        y_abs_norm = np.clip(y_abs_norm, 0, None)
        Y_norm = y_abs_norm * np.exp(1j * y_theta)
        targets = np.stack([np.real(Y_norm), np.imag(Y_norm)], axis=-1).astype(np.float32)

        self.y_abs_max = y_abs_max

        self.x_abs_seq = []
        self.theta_seq = []
        self.target_seq = []

        N = len(x_abs)
        for i in range(N - seq_len):
            self.x_abs_seq.append(x_abs[i:i+seq_len])
            self.theta_seq.append(x_theta[i:i+seq_len])
            self.target_seq.append(targets[i:i+seq_len])

        self.x_abs_seq = torch.from_numpy(np.array(self.x_abs_seq))
        self.theta_seq = torch.from_numpy(np.array(self.theta_seq))
        self.target_seq = torch.from_numpy(np.array(self.target_seq))

    def __len__(self):
        return self.x_abs_seq.shape[0]

    def __getitem__(self, idx):
        return (
            self.x_abs_seq[idx],
            self.theta_seq[idx],
            self.target_seq[idx]
        )

class TrainModel(nn.Module):
    def __init__(self, seq_len=10, hidden_size=64, n_epochs=30, batch_size=64, mat_path='U_and_U_ideal_for_pjanet.mat', learning_rate=5e-3, invert=False, norm_func=max_norm):
        super(TrainModel, self).__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.mat_path = mat_path
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print('Loading dataset...')
        self.dataset = PGJanetSequenceDataset(mat_path, seq_len=seq_len, invert=invert, norm_func=norm_func)
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        # Initialize model and optimizer
        self.model = PGJanetRNN(hidden_size=hidden_size).to(self.device)
        self.loss_function = nMSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=1, factor=0.7, threshold=1e-4, min_lr=1e-7)

        self.learning_rates = []
        self.epoch_losses_list = []
        self.val_losses_list = []

    # Validation split ratio
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
            # Loop through each batch of training data
            for batch_x_abs, batch_theta, batch_targets in self.train_loader:
                                # Remove the last dimension: (batch, seq_len, 1) -> (batch, seq_len)
                batch_x_abs = batch_x_abs.to(self.device).squeeze(-1)
                                # Same squeezing for phase
                batch_theta = batch_theta.to(self.device).squeeze(-1)
                batch_targets = batch_targets.to(self.device)

                outputs = self.model(batch_x_abs, batch_theta)
                loss = self.loss_function(outputs, batch_targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= len(self.train_loader)
            self.epoch_losses_list.append(epoch_loss)

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

                        # Adjust learning rate based on training loss
            self.scheduler.step(epoch_loss)
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])

            print(f"Epoch {epoch+1}, Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {self.optimizer.param_groups[0]['lr']:.6e}")

        print('Training complete.')
        loss_db = 10 * np.log10(np.array(self.epoch_losses_list) + 1e-12)
        val_db = 10 * np.log10(np.array(self.val_losses_list) + 1e-12)
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
                # Save training/validation curves to PNG
        fig.savefig('epoch_loss.png')

    def save_model(self, path='pg_janet_rnn_max.pth'):
        print(f'Saving model to {path}...')
        torch.save(self.model.state_dict(), path)
        print('Model saved.')

if __name__ == "__main__":
    # -----------------------------
    # Load dataset and split into train/validation
    # -----------------------------
    val_ratio = 0.25  # 25% for validation
    seq_len = 16
    hidden_size = 32
    stats_path = 'pg_janet_stats.npz'

        # Load dataset (TX1_BB -> TX1_SISO mapping by default)
    dataset = PGJanetSequenceDataset( seq_len=seq_len, invert=False, norm_func=max_norm)
    n_val = int(len(dataset) * val_ratio)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    real_loader = DataLoader(dataset, batch_size=1, shuffle=False)

        # Initialize and train the model with configuration options
    train_model = TrainModel(seq_len=seq_len, hidden_size=hidden_size, mat_path='U_and_U_ideal_for_pjanet.mat', invert=False, norm_func=max_norm)
    train_model.train()
    train_model.save_model('pg_janet_rnn_ilc_direct.pth')
