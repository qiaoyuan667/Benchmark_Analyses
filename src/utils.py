import logging
import matplotlib.pyplot as plt


def setup_logger(log_file, level=logging.INFO):
    handlers = [logging.StreamHandler()]

    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True
    )


def str2bool(arg):
    if arg in ['1', 'true', 'True']:
        return True
    elif arg in ['0', 'false', 'False']:
        return False
    else:
        return None


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
            'total_active_set': set()
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

        hist = {}
        for v in self.current['active_values']:
            bin_idx = int(v // 0.1)
            hist[bin_idx] = hist.get(bin_idx, 0) + 1

        stats = {
            # -------- loss --------
            'total_loss':  sum(self.current['loss_total']),
            'loss_recon': sum(self.current['loss_recon']),
            'loss_sparse': sum(self.current['loss_sparse']),
            # -------- latent --------
            'avg_active': sum(self.current['avg_active_list']) / len(self.current['avg_active_list']),
            'total_active': len(self.current['total_active_set']),
            # -------- act dist --------
            'active_distribution': hist
        }

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