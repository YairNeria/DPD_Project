import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.io import loadmat

class FC_Model(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=40, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class Signal_Dataset(Dataset):
    def __init__(self, mat_path='for_DPD.mat', M=3):
        mat = loadmat(mat_path, squeeze_me=True)
        X = mat['TX1_BB']
        Y = mat['TX1_SISO']
        self.M = M
        inputs, targets = [], []
        for n in range(M-1, len(X)):
            x_taps = X[n-M+1:n+1]
            if len(x_taps) < M:
                continue
            input_vec = np.empty(2*M, dtype=np.float32)
            input_vec[0::2] = x_taps.real
            input_vec[1::2] = x_taps.imag
            inputs.append(input_vec)
            targets.append([Y[n].real, Y[n].imag])
        self.x_data = torch.from_numpy(np.stack(inputs)).float()
        self.y_data = torch.from_numpy(np.stack(targets)).float()
        self.n_samples = self.x_data.shape[0]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.n_samples
#כככ
class FC_Trainer:
    def __init__(self, dataset, model=None, lr=1e-3, batch_size=64, n_epochs=20):
        self.dataset = dataset
        self.model = model if model is not None else FC_Model(input_dim=2*dataset.M)
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.n_epochs = n_epochs
    def train(self):
        for epoch in range(self.n_epochs):
            for batch_in, batch_tar in self.loader:
                outputs = self.model(batch_in)
                loss = self.loss_function(outputs, batch_tar)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")
    def save_model(self, path='fc_model.pth'):
        torch.save(self.model.state_dict(), path)
    def load_model(self, path='fc_model.pth'):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

# Example usage (uncomment to run as script):
# if __name__ == "__main__":
#     dataset = Signal_Dataset(M=3)
#     trainer = FC_Trainer(dataset)
#     trainer.train()
#     trainer.save_model()
