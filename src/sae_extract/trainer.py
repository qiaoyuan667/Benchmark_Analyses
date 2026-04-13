import logging
import torch
from src.utils import StatsTracker
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F


class SAETrainer:
    def __init__(self, sae_model, dataset, l1_weight, batch_size, lr, epochs, device, do_val, do_test, min_activation, early_stop, auk_ratio, aux_weight, dead_steps, verbose=True):
        self.model = sae_model
        self.full_dataset = dataset
        self.best_model_state = None

        self.device = device
        self.model.to(device)
        self.verbose = verbose

        self.l1_weight = l1_weight
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.last_epoch = epochs
        self.best_epoch = None

        self.early_stop = early_stop

        self.do_val = do_val
        self.do_test = do_test

        self.activation_threshold = 1e-3

        # self.training_data = Subset(self.full_dataset, indices=self.full_dataset.training_indices)
        # self.val_data = Subset(self.full_dataset, indices=self.full_dataset.val_indices)
        # self.test_data = Subset(self.full_dataset, indices=self.full_dataset.test_indices)

        self.training_tracker = StatsTracker(min_activation)
        self.val_tracker = StatsTracker(min_activation)
        self.test_tracker = StatsTracker(min_activation)

        # avoid dead latents
        self.steps_since_activation = torch.zeros(self.model.hidden_dim, device=self.device)
        self.dead_steps = dead_steps
        self.aux_ratio = auk_ratio
        self.aux_weight = aux_weight

    def train(self, use_full):
        self.model.train()
        if use_full:
            train_loader = DataLoader(self.full_dataset, batch_size=self.batch_size, shuffle=True)
            self.do_val = False
        else:
            train_loader = DataLoader(self.full_dataset.get_split('train'), batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        logging.info(f'Start training on {"full" if use_full else "train"} with batch_size={self.batch_size}, lr={self.lr}, epochs={self.epochs}, l1_weight={self.l1_weight}, aux_weight={self.aux_weight}, dead_steps={self.dead_steps}, aux_ratio={self.aux_ratio}, device={self.device}')

        best_loss = float('inf')
        patience = 5
        no_improve_count = 0

        for epoch in range(self.epochs):
            total_loss = 0
            total_recon = 0
            total_sparse = 0
            self.training_tracker.init_stats()
            for batch_id, x_batch in enumerate(train_loader):

                optimizer.zero_grad()
                pre_act, z, x_hat = self.model.forward(x_batch['response'].to(self.device))

                # track dead latents
                self.steps_since_activation += 1
                fired = (z.abs() > self.activation_threshold).any(dim=0)
                if fired.numel() > 0:
                    self.steps_since_activation[fired] = 0

                loss, recon, sparse = self.loss_fn(x_batch['response'].to(self.device), x_hat, z, pre_act=pre_act, mask=x_batch['mask'].to(self.device))

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
                self.eval(split='full' if use_full else 'val')
                logging.info(f"Epoch {epoch+1}/{self.epochs}, Val recon loss: {self.val_tracker.get_last()['loss_recon']:.2f}" 
                             f" | Avg active: {self.val_tracker.get_last()['avg_active']:.2f}" 
                             f" | Total active: {self.val_tracker.get_last()['total_active']}/{self.model.hidden_dim}")

            # best epoch (on train or val)
            cur_tracker = self.val_tracker if self.do_val else self.training_tracker
            if cur_tracker.get_last()['loss_recon'] < best_loss:
                best_loss = cur_tracker.get_last()['loss_recon']
                no_improve_count = 0
                self.best_model_state = {
                    k: v.cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
                self.best_epoch = epoch + 1
                logging.info(f"Best model {epoch+1} on {'Val' if self.do_val else 'Full'} ! Reconstruction loss: {best_loss:.4f}")
            else:
                no_improve_count += 1
                logging.info(f"No improvement for {no_improve_count} epochs")

            if no_improve_count >= patience:
                logging.info(f"Early stopping at epoch {epoch + 1}")
                self.last_epoch = epoch + 1
                break

        logging.info(f'Training finished, last epoch {self.last_epoch}, best epoch {self.best_epoch}')

    def load_best(self):
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

    def loss_fn(self, x, x_hat, z, pre_act, mask):
        # mask: 1 = ignore, 0 = valid
        valid_mask = (mask == 0)

        # reconstruction loss (masked MSE)
        recon = F.mse_loss(x_hat, x, reduction='none')
        recon = recon * valid_mask
        recon_loss = recon.sum() / valid_mask.sum().clamp(min=1)

        # sparsity (L1 on latent)
        sparsity_loss = torch.mean(torch.abs(z))

        # aux_loss
        # 1. identify dead neurons
        dead_mask = (self.steps_since_activation > self.dead_steps)
        # 2. select from dead neurons (use pre_act!)
        aux_k = int(self.aux_ratio * self.model.hidden_dim)
        dead_pre_act = pre_act.masked_fill(~dead_mask, -1e9)
        aux_vals, aux_idx = torch.topk(dead_pre_act, aux_k, dim=-1)
        aux_vals = F.relu(aux_vals)

        # 3. build aux activation
        aux_act = torch.zeros_like(z)
        aux_act.scatter_(-1, aux_idx, aux_vals)
        # 4. residual (detach main path!!)
        residual = x - x_hat.detach()
        # 5. reconstruct residual
        aux_recon = self.model.decode(aux_act)
        aux_loss = F.mse_loss(aux_recon, residual)

        total_loss = recon_loss + self.l1_weight * sparsity_loss + self.aux_weight * aux_loss

        return total_loss, recon_loss, sparsity_loss

    def eval(self, split):
        self.model.eval()
        eval_dataset = self.full_dataset.get_split(split)
        tracker = StatsTracker(self.activation_threshold)
        data_loader = DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=False)
        tracker.init_stats()

        id_all = []
        z_all = []

        with torch.no_grad():
            for _, data in enumerate(data_loader):
                id_all.extend(data['item_id'].tolist())
                pre_act, z, x_hat = self.model(data['response'].to(self.device))
                z_all.append(z.cpu())

                total_loss, recon_loss, sparse_loss = self.loss_fn(x=data['response'].to(self.device), x_hat=x_hat, z=z, mask=data['mask'].to(self.device), pre_act=pre_act)
                tracker.update_stats(total_loss, recon_loss, sparse_loss, z)
        tracker.summary_stats()

        z_all = torch.cat(z_all, dim=0).tolist()

        assert len(id_all) == len(z_all), f'mismatched output, {len(id_all)} ids, {len(z_all)} outputs'

        return {
            'features': {id_all[i]: z_all[i] for i in range(len(z_all))},
            'stats': tracker.get_last()
        }

    def save(self, save_path, save_best=True):
        epoch_id = self.best_epoch if save_best else self.last_epoch
        checkpoint = {
            # -------- model --------
            "model_state": self.best_model_state if save_best else self.model.state_dict() ,
            # -------- config --------
            "config": {
                "l1_weight": self.l1_weight,
                "batch_size": self.batch_size,
                "lr": self.lr,
                "epochs": epoch_id,
                "activation_threshold": self.activation_threshold,
            },
            "training_stats": self.training_tracker.stats_all,
            "val_stats": self.val_tracker.stats_all,
            "test_stats": self.test_tracker.stats_all,
        }
        torch.save(checkpoint, save_path)
        if self.verbose:
            logging.info(f"{'Best' if save_best else 'Last'} checkpoint {epoch_id} saved to {save_path}")

    def load(self, load_path):
        checkpoint = torch.load(load_path, map_location=self.device)

        # -------- model --------
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)

        # -------- config--------
        config = checkpoint.get("config", {})
        self.l1_weight = config.get("l1_weight", self.l1_weight)
        self.batch_size = config.get("batch_size", self.batch_size)
        self.lr = config.get("lr", self.lr)
        self.epochs = config.get("epochs", self.epochs)
        self.activation_threshold = config.get("activation_threshold", self.activation_threshold)

        # -------- trackers --------
        self.training_tracker.stats_all = checkpoint.get("training_stats", [])
        self.val_tracker.stats_all = checkpoint.get("val_stats", [])
        self.test_tracker.stats_all = checkpoint.get("test_stats", [])

        if self.verbose:
            logging.info(f"Checkpoint loaded from {load_path}")