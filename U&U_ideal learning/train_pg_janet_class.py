import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from scipy.io import loadmat
from pg_janet import PGJanetRNN
import matplotlib.pyplot as plt


#TODO - think about normalize x and y independently 

class nMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, y_pred, y_true):
        mse = torch.sum((y_pred - y_true) ** 2)
        denom = torch.sum(y_true ** 2) + 1e-8
        return mse / denom

def std_norm(x, mean=None, std=None):
    if mean is None:
        mean = x.mean()
    if std is None:
        std = x.std()
    return (x - mean) / (std + 1e-8)

def max_norm(x, max_val=None):
    if max_val is None:
        max_val = np.max(np.abs(x))
    return x / (max_val + 1e-8) 

    def norm(x, x_m, x_s):
        return (x - x_m) / (x_s + 1e-8)


class PGJanetSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, mat_path='for_DPD.mat', seq_len=12, invert=False, Norm_func=std_norm, save_stats_path=None):
        # Load data
        mat = loadmat(mat_path, squeeze_me=True)
        if not invert:
            X = mat['TX1_BB']     # Original signal
            Y = mat['TX1_SISO']   # Distorted signal
        else:
            X = mat['TX1_SISO']   # Distorted signal
            Y = mat['TX1_BB']     # Original signal
        self.norm_func = Norm_func  # Normalization function
        # Extract amplitude and phase from input signal
        amplitudes = np.abs(X).astype(np.float32).reshape(-1, 1)
        amplitudes = self.norm_func(amplitudes)  # Normalize input amplitude
        phases = np.angle(X).astype(np.float32)

        # Stack real and imaginary parts of output signal
        targets = np.stack([Y.real, Y.imag], axis=-1).astype(np.float32).reshape(-1, 2)

        # Normalize target amplitude (magnitude of target complex signal) separately
        target_complex = targets[..., 0] + 1j * targets[..., 1]
        target_amp = np.abs(target_complex)
        target_amp = self.norm_func(target_amp)  # Normalize target amplitude
        #build the target signal with the now normalized amplitude and same phase 
        target_phase = np.angle(target_complex)
        targets = np.stack([target_amp * np.cos(target_phase), target_amp * np.sin(target_phase)], axis=-1) 
        
        # Store means and stds for later use (e.g., for denormalization in evaluation)
        self.amp_mean = amplitudes.mean()
        self.amp_std = amplitudes.std()
        self.target_amp_mean = target_amp.mean()
        self.target_amp_std = target_amp.std()


        N = len(amplitudes)
        self.x_abs_seq = []
        self.theta_seq = []
        self.target_seq = []

        for i in range(N - seq_len):
            self.x_abs_seq.append(amplitudes[i:i+seq_len])
            self.theta_seq.append(phases[i:i+seq_len])
            self.target_seq.append(targets[i:i+seq_len])

        self.x_abs_seq = torch.from_numpy(np.array(self.x_abs_seq))         # shape: (samples, seq_len)
        self.theta_seq = torch.from_numpy(np.array(self.theta_seq))         # shape: (samples, seq_len)
        self.target_seq = torch.from_numpy(np.array(self.target_seq))       # shape: (samples, seq_len, 2)

    def __len__(self):
        return self.x_abs_seq.shape[0]

    def __getitem__(self, idx):
        return (
            self.x_abs_seq[idx],     # shape: (seq_len, 1)
            self.theta_seq[idx],     # shape: (seq_len, 1)
            self.target_seq[idx]     # shape: (seq_len, 2)
        )

class TrainModel(nn.Module):
    def __init__(self, seq_len=10, hidden_size=64, n_epochs=30, batch_size=64, mat_path='for_DPD.mat', learning_rate=5e-3, save_stats_path=None):
        super(TrainModel, self).__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.mat_path = mat_path
        self.learning_rate = learning_rate

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print('Loading dataset...')
        self.dataset = PGJanetSequenceDataset(mat_path, seq_len=seq_len, invert=False, Norm_func=max_norm, save_stats_path=save_stats_path)
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        # Model + optimizer
        self.model = PGJanetRNN(hidden_size=hidden_size).to(self.device)
        self.loss_function = nMSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=1, factor=0.7, threshold=1e-4, min_lr=1e-7)

        self.learning_rates = []
        self.epoch_losses_list = []

    def train(self):
        print('Starting training...')

        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            for batch_x_abs, batch_theta, batch_targets in self.loader:
                batch_x_abs = batch_x_abs.to(self.device)
                batch_theta = batch_theta.to(self.device)
                batch_targets = batch_targets.to(self.device)

                # Squeeze last dimension to get (batch, seq_len) before model
                batch_x_abs = batch_x_abs.squeeze(-1)
                batch_theta = batch_theta.squeeze(-1)
                outputs = self.model(batch_x_abs, batch_theta)
                loss = self.loss_function(outputs, batch_targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= len(self.loader)
            self.scheduler.step(epoch_loss)
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
            self.epoch_losses_list.append(epoch_loss)

            print(f"Epoch {epoch+1}, Loss: {epoch_loss:.6f}, LR: {self.optimizer.param_groups[0]['lr']:.6e}")

        print('Training complete.')

        # Plot the loss (in dB) and learning rate in each epoch
        loss_db = 10 * np.log10(np.array(self.epoch_losses_list) + 1e-12)
        fig, ax1 = plt.subplots()
        ax1.plot(loss_db, 'b-', label='Loss (dB)')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (dB)', color='b')
        ax2 = ax1.twinx()
        ax2.plot(self.learning_rates, 'r--', label='Learning Rate')
        ax2.set_ylabel('Learning Rate', color='r')
        plt.title('Training Loss (dB) and Learning Rate per Epoch')
        fig.tight_layout()
        plt.show()
        fig.savefig('epoch_loss.png')

    def save_model(self, path='pg_janet_rnn.pth'):
        print(f'Saving model to {path}...')
        torch.save(self.model.state_dict(), path)
        print('Model saved.')

if __name__ == "__main__":
    # -----------------------------
    # Load dataset and split into train/validation
    # -----------------------------
    val_ratio = 0.25  # 25% for validation
    seq_len = 10
    hidden_size = 64
    stats_path = 'pg_janet_stats.npz'  # Path to save normalization statistics
    # Create the dataset with a sequence length of 10 and save stats
    dataset = PGJanetSequenceDataset('for_DPD.mat', seq_len=seq_len, invert=False, Norm_func=max_norm, save_stats_path=stats_path)
    n_val = int(len(dataset) * val_ratio)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    # Create DataLoaders for training and validation
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    real_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    train_model = TrainModel(seq_len=seq_len, hidden_size=hidden_size, save_stats_path=stats_path)
    train_model.train()
    train_model.save_model('pg_janet_rnn_max.pth')






