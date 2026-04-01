import pickle
import sys
import os
import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from src.data import ResponseDataset
from pathlib import Path
from src.config import setup_logger, local_path
import matplotlib.pyplot as plt


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, tied_weights=True):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.tied_weights = tied_weights

        # encoder
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)

        # decoder
        if tied_weights:
            self.decoder_bias = nn.Parameter(torch.zeros(input_dim))
        else:
            self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)

    # -------- forward --------
    def encode(self, x):
        z = F.relu(self.encoder(x))
        return z

    def decode(self, z):
        if self.tied_weights:
            # tied weights: decoder = encoder^T
            return F.linear(z, self.encoder.weight.t(), self.decoder_bias)
        else:
            return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    # -------- inference --------
    def transform(self, data, batch_size):
        """Get latent representation z for all data"""
        self.eval()
        loader = DataLoader(data, batch_size=batch_size)

        z_all = []
        with torch.no_grad():
            for data in loader:
                z = self.encode(data['response'].to(self.device))
                z_all.append(z.cpu())

        return torch.cat(z_all, dim=0)

    def reconstruct(self, data):
        """Reconstruct input"""
        self.eval()
        with torch.no_grad():
            data = torch.tensor(data, dtype=torch.float32).to(self.device)
            x_hat, _ = self.forward(data)
        return x_hat.cpu()

    def get_dictionary(self):
        """Return learned features (decoder directions)"""
        if self.tied_weights:
            return self.encoder.weight.detach().cpu()
        else:
            return self.decoder.weight.detach().cpu()


class StatsTracker:
    def __init__(self, act_threshold):
        self.act_threshold = act_threshold
        self.stats_all = []
        self.current = None

    def init_stats(self):
        self.current = {
            'loss_total': [],
            'loss_recon': [],
            'loss_sparse': [],
            'avg_active_list': [],   # list[int]
            'active_values': [],     # list[float]
            'total_active_set': set()  # 用 set 记录激活的 latent index
        }

    def update_stats(self, loss, recon, sparse, z):
        assert self.current is not None, "Call init_stats() first"

        # -------- loss --------
        self.current['loss_total'].append(loss.item())
        self.current['loss_recon'].append(recon.item())
        self.current['loss_sparse'].append(sparse.item())

        # -------- latent stats --------
        z = z.detach().cpu()

        active = (z.abs() > self.act_threshold)

        # (1) per-input active count → int
        active_per_input = active.sum(dim=1).tolist()  # list[int]
        self.current['avg_active_list'].extend(active_per_input)

        # (2) activation values → float list
        self.current['active_values'].extend(z.view(-1).tolist())

        # (3) global active latent → 用 index set
        active_indices = active.any(dim=0).nonzero(as_tuple=False).view(-1).tolist()
        self.current['total_active_set'].update(active_indices)

    def summary_stats(self):
        assert self.current is not None

        stats = self.current

        # -------- loss --------
        stats['loss_total'] = sum(stats['loss_total'])
        stats['loss_recon'] = sum(stats['loss_recon'])
        stats['loss_sparse'] = sum(stats['loss_sparse'])

        # -------- latent --------
        stats['avg_active'] = sum(stats['avg_active_list']) / len(stats['avg_active_list'])
        stats['total_active'] = len(stats['total_active_set'])

        # -------- clear --------
        del stats['avg_active_list']
        del stats['total_active_set']

        # -------- store --------
        self.stats_all.append(stats)
        self.current = None

    # -------- utils --------
    def get_last(self):
        return self.stats_all[-1] if self.stats_all else None

    def get_history(self):
        return self.stats_all

    def plot_loss(self):
        stats = self.stats_all
        epochs = list(range(1, len(stats) + 1))

        loss_total = [s['loss_total'] for s in stats]
        loss_recon = [s['loss_recon'] for s in stats]
        loss_sparse = [s['loss_sparse'] for s in stats]

        plt.figure()
        plt.plot(epochs, loss_total, linestyle='-')
        plt.plot(epochs, loss_recon, linestyle='--')
        plt.plot(epochs, loss_sparse, linestyle=':')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss over Epochs")
        plt.legend(["Total", "Reconstruction", "Sparsity"])
        plt.show()

    def plot_avg_active(self):
        stats = self.stats_all
        epochs = list(range(1, len(stats) + 1))

        avg_active = [s['avg_active'] for s in stats]

        plt.figure()
        plt.plot(epochs, avg_active)
        plt.xlabel("Epoch")
        plt.ylabel("Avg Active per Input")
        plt.title("Average Active Latents")
        plt.show()

    def plot_total_active(self):
        stats = self.stats_all
        epochs = list(range(1, len(stats) + 1))

        total_active = [s['total_active'] for s in stats]

        plt.figure()
        plt.plot(epochs, total_active)
        plt.xlabel("Epoch")
        plt.ylabel("Number of Active Latents")
        plt.title("Total Active Latents")
        plt.show()

    def plot_activation_distribution(self):
        last_stats = self.stats_all[-1]
        active_values = last_stats['active_values']
        plt.figure()
        plt.hist(active_values, bins=50)
        plt.xlabel("Latent Activation Value")
        plt.ylabel("Frequency")
        plt.title("Activation Distribution (Last Epoch)")
        plt.show()


class SAETrainer:
    def __init__(self, sae_model, dataset, l1_weight, batch_size, lr, epochs, device, do_val, do_test, min_activation, verbose=True):
        self.model = sae_model
        self.dataset = dataset

        self.device = device
        self.model.to(device)
        self.verbose = verbose

        self.l1_weight = l1_weight
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs

        self.do_val = do_val
        self.do_test = do_test

        self.activation_threshold = 1e-3

        self.training_data = Subset(self.dataset, indices=self.dataset.training_indices)
        self.val_data = Subset(self.dataset, indices=self.dataset.val_indices)
        self.test_data = Subset(self.dataset, indices=self.dataset.test_indices)

        self.training_tracker = StatsTracker(min_activation)
        self.val_tracker = StatsTracker(min_activation)
        self.test_tracker = StatsTracker(min_activation)

    def train(self):
        self.model.train()

        train_loader = DataLoader(self.training_data, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        logging.info(f'Start training with batch_size={self.batch_size}, lr={self.lr}, epochs={self.epochs}, l1_weight={self.l1_weight}, device={self.device}')
        for epoch in range(self.epochs):
            total_loss = 0
            total_recon = 0
            total_sparse = 0
            self.training_tracker.init_stats()
            for batch_id, x_batch in enumerate(train_loader):

                optimizer.zero_grad()
                x_hat, z = self.model.forward(x_batch['response'].to(self.device))
                loss, recon, sparse = self.loss_fn(x_batch['response'].to(self.device), x_hat, z, mask=x_batch['mask'].to(self.device))

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_recon += recon.item()
                total_sparse += sparse.item()

                self.training_tracker.update_stats(loss, recon, sparse, z)
                #
                # if self.verbose:
                #     logging.info(f"Epoch {epoch+1}/{self.epochs} | batch {batch_id+1}/{len(train_loader)}  | Loss: {loss:.4f} | Recon: {recon:.4f} | Sparse: {sparse:.4f}.")

            self.training_tracker.summary_stats()
            logging.info(f"Epoch {epoch+1}/{self.epochs} | Total_loss: {total_loss} | Avg active: {self.training_tracker.get_last()['avg_active']:.2f} | Total active: {self.training_tracker.get_last()['total_active']}/{self.model.hidden_dim}")
            if self.do_val:
                self.eval(split='val')
                logging.info(f"Epoch {epoch+1}/{self.epochs}, Val recon loss: {self.val_tracker.get_last()['loss_recon']:.2f}" 
                             f" | Avg active: {self.val_tracker.get_last()['avg_active']:.2f}" 
                             f" | Total active: {self.val_tracker.get_last()['total_active']}/{self.model.hidden_dim}")

        if self.do_test:
            self.eval(split='test')
            logging.info(f"Test recon loss: {self.test_tracker.get_last()['loss_recon']:.2f}"
                         f" | Avg active: {self.test_tracker.get_last()['avg_active']:.2f}"
                         f" | Total active: {self.test_tracker.get_last()['total_active']}/{self.model.hidden_dim}")

    def loss_fn(self, x, x_hat, z, mask):
        # mask: 1 = ignore, 0 = valid
        valid_mask = (mask == 0)

        # reconstruction loss (masked MSE)
        recon = F.mse_loss(x_hat, x, reduction='none')
        recon = recon * valid_mask
        recon_loss = recon.sum() / valid_mask.sum().clamp(min=1)

        # sparsity (L1 on latent)
        sparsity_loss = torch.mean(torch.abs(z))
        total_loss = recon_loss + self.l1_weight * sparsity_loss

        return total_loss, recon_loss, sparsity_loss

    def eval(self, split):
        assert split in ['train', 'val', 'test']
        self.model.eval()
        if split == 'val':
            eval_dataset = self.val_data
            tracker = self.val_tracker
        else:
            eval_dataset = self.test_data
            tracker = self.test_tracker

        data_loader = DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=False)
        tracker.init_stats()
        with torch.no_grad():
            for _, data in enumerate(data_loader):
                x_hat, z = self.model(data['response'].to(self.device))
                total_loss, recon_loss, sparse_loss = self.loss_fn(x=data['response'].to(self.device), x_hat=x_hat, z=z, mask=data['mask'])
                tracker.update_stats(total_loss, recon_loss, sparse_loss, z)
        tracker.summary_stats()

    def save(self, save_path):
        checkpoint = {
            # -------- model --------
            "model_state": self.model.state_dict(),
            # -------- config --------
            "config": {
                "l1_weight": self.l1_weight,
                "batch_size": self.batch_size,
                "lr": self.lr,
                "epochs": self.epochs,
                "activation_threshold": self.activation_threshold,
            },
            # -------- trackers（现在都是纯 Python）--------
            "training_tracker": self.training_tracker.stats_all,
            "val_tracker": self.val_tracker.stats_all,
            "test_tracker": self.test_tracker.stats_all,
        }
        torch.save(checkpoint, save_path)
        if self.verbose:
            logging.info(f"Checkpoint saved to {save_path}")

    def load(self, load_path):
        checkpoint = torch.load(load_path, map_location=self.device)

        # -------- model --------
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)

        # -------- config（可选恢复）--------
        config = checkpoint.get("config", {})
        self.l1_weight = config.get("l1_weight", self.l1_weight)
        self.batch_size = config.get("batch_size", self.batch_size)
        self.lr = config.get("lr", self.lr)
        self.epochs = config.get("epochs", self.epochs)
        self.activation_threshold = config.get("activation_threshold", self.activation_threshold)

        # -------- trackers --------
        self.training_tracker.stats_all = checkpoint.get("training_tracker", [])
        self.val_tracker.stats_all = checkpoint.get("val_tracker", [])
        self.test_tracker.stats_all = checkpoint.get("test_tracker", [])

        if self.verbose:
            logging.info(f"Checkpoint loaded from {load_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model setting
    parser.add_argument('--hidden_dim', type=int)
    parser.add_argument('--l1_weight', type=float)
    parser.add_argument('--tie_weights', type=bool, default=True)
    parser.add_argument('--min_activation', type=float)
    # training setting
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--do_val', type=bool, )
    parser.add_argument('--do_test', type=bool)
    # path
    parser.add_argument('--model_save_path', type=Path, )
    parser.add_argument('--model_load_path', type=Path)
    parser.add_argument('--log_file', type=Path, default='log.txt')

    args = parser.parse_args()

    setup_logger(log_file=args.log_file)

    # dataset
    if os.path.exists(local_path['cached_response_matrix']):
        with open(local_path['cached_response_matrix'], 'rb') as f_data:
            response_dataset = pickle.load(f_data)
    else:
        response_dataset = ResponseDataset(local_path['raw_response_matrix'], min_coverage=0.8)
        with open(local_path['cached_response_matrix'], "wb") as f_data:
            pickle.dump(response_dataset, f_data)

    # model
    model = SparseAutoencoder(input_dim=len(response_dataset.clean_subjects), hidden_dim=args.hidden_dim)

    # train
    trainer = SAETrainer(
        sae_model=model,
        dataset=response_dataset,
        l1_weight=args.l1_weight,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=torch.device('gpu') if torch.cuda.is_available() else torch.device('cpu'),
        do_val=args.do_val,
        do_test=args.do_test,
        min_activation=args.min_activation
    )
    trainer.train()

    # plot
    trainer.training_tracker.plot_loss()
    trainer.training_tracker.plot_avg_active()
    trainer.training_tracker.plot_total_active()
    trainer.training_tracker.plot_activation_distribution()

    if args.do_val:
        trainer.val_tracker.plot_loss()
        trainer.val_tracker.plot_avg_active()
        trainer.val_tracker.plot_total_active()
        trainer.val_tracker.plot_activation_distribution()
    if args.do_test:
        trainer.test_tracker.plot_loss()
        trainer.test_tracker.plot_avg_active()
        trainer.test_tracker.plot_total_active()
        trainer.test_tracker.plot_activation_distribution()
