import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class LassoSparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, tied_weights=True):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.tied_weights = tied_weights
        # encoder
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)

        # decoder
        if not tied_weights:
            self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)

        self.input_bias = nn.Parameter(torch.zeros(input_dim))

    # -------- forward --------
    def encode(self, x):
        # W_enc(x - b_pre) + b_enc
        pre_act = self.encoder(x-self.input_bias)
        z = F.relu(pre_act)
        return pre_act, z

    def decode(self, z):
        if self.tied_weights:
            # tied weights: decoder = encoder^T
            return F.linear(z, self.encoder.weight.t(), self.input_bias)
        else:
            return self.decoder(z)

    def forward(self, x):
        pre_act, z = self.encode(x)
        x_hat = self.decode(z)
        return pre_act, z, x_hat

    # -------- inference --------
    def transform(self, data, batch_size):
        """Get latent representation z for all data"""
        self.eval()
        loader = DataLoader(data, batch_size=batch_size, shuffle=False)

        id_all = []
        z_all = []
        with torch.no_grad():
            for data in loader:
                id_all.extend(data['item_id'])
                z = self.encode(data['response'].to(self.device))
                z_all.append(z.cpu())
        z_all = torch.cat(z_all, dim=0).tolist()

        assert len(id_all) == len(z_all), 'mismatched output'
        return {id_all[i]: z_all[i] for i in range(len(z_all))}

    def reconstruct(self, data):
        """Reconstruct input"""
        self.eval()
        with torch.no_grad():
            data = torch.tensor(data, dtype=torch.float32).to(self.device)
            pre_act, z, x_hat = self.forward(data)
        return x_hat.cpu()

    def get_dictionary(self):
        """Return learned features (decoder directions)"""
        if self.tied_weights:
            return self.encoder.weight.detach().cpu()
        else:
            return self.decoder.weight.detach().cpu()


class TopKSparseAutoEncoder(nn.Module):
    def __init__(self):
        super(TopKSparseAutoEncoder, self).__init__()
        pass

    def forward(self):
        pass

    def encode(self):
        pass

    def decode(self):
        pass

