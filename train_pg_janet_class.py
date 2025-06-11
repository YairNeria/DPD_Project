import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy.io import loadmat
from pg_janet import PGJanetRNN

class nMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, y_pred, y_true):
        mse = torch.sum((y_pred - y_true) ** 2)
        denom = torch.sum(y_true ** 2) + 1e-8
        return mse / denom

class PGJanetSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, mat_path='for_DPD.mat', seq_len=10):
        mat = loadmat(mat_path, squeeze_me=True)
        X = mat['TX1_BB']
        Y = mat['TX1_SISO']
        amplitudes = np.abs(X).astype(np.float32)
        amplitudes /= np.max(amplitudes)  # Normalize to [0, 1]
        phases = np.angle(X).astype(np.float32)
        targets = np.stack([Y.real, Y.imag], axis=-1).astype(np.float32)
        N = len(amplitudes)
        self.x_abs_seq = []
        self.theta_seq = []
        self.target_seq = []
        for i in range(N - seq_len):
            self.x_abs_seq.append(amplitudes[i:i+seq_len])
            self.theta_seq.append(phases[i:i+seq_len])
            self.target_seq.append(targets[i:i+seq_len])
        self.x_abs_seq = torch.from_numpy(np.array(self.x_abs_seq))
        self.theta_seq = torch.from_numpy(np.array(self.theta_seq))
        self.target_seq = torch.from_numpy(np.array(self.target_seq))
    def __len__(self):
        return self.x_abs_seq.shape[0]
    def __getitem__(self, idx):
        return self.x_abs_seq[idx].unsqueeze(-1), self.theta_seq[idx].unsqueeze(-1), self.target_seq[idx]

class TrainModel:
    def __init__(self, seq_len=12, hidden_size=16, n_epochs=30, batch_size=64, mat_path='for_DPD.mat'):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.mat_path = mat_path
        print('Loading dataset...')
        self.dataset = PGJanetSequenceDataset(mat_path, seq_len=seq_len)
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        self.model = PGJanetRNN(hidden_size=hidden_size)
        self.loss_function = nMSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=2, factor=0.7, threshold=5e-5, min_lr=1e-7)
        self.learning_rates = []
        self.epoch_losses_list = []

    def train(self):
        print('Starting training...')
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            for batch_x_abs, batch_theta, batch_targets in self.loader:
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

    def save_model(self, path='pg_janet_rnn.pth'):
        print(f'Saving model to {path}...')
        torch.save(self.model.state_dict(), path)
        print('Model saved.')

if __name__ == "__main__":
    trainer = TrainModel()
    trainer.train()
    trainer.save_model()
