import logging
import sys
import os
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, Subset


class ResponseDataset(Dataset):
    def __init__(self, input_file, min_coverage, do_normalize=False, split_ratio=(0.8, 0.1, 0.1), seed=42):
        super().__init__()

        self.min_coverage = min_coverage
        logging.info('Reading data from {}'.format(input_file))
        df = pd.read_csv(input_file)
        self.full_models = df.iloc[:, 0].tolist()
        self.full_items = [int(item_id.split('_')[1]) for item_id in df.columns.tolist()[1:]]
        self.full_response_matrix = df.iloc[:, 1:].astype(np.float32).to_numpy()

        # prune
        self.clean_response_matrix, self.drop_rows, self.drop_columns = self.prune()

        self.clean_subjects = [
            self.full_models[idx]
            for idx in range(len(self.full_models))
            if idx not in self.drop_rows
        ]
        self.clean_items = [
            self.full_items[idx]
            for idx in range(len(self.full_items))
            if idx not in self.drop_columns
        ]

        if do_normalize:
            self.normalize()

        self.training_indices, self.val_indices, self.test_indices = self.build_split(split_ratio, seed)

    def build_split(self, split_ratio, seed):
        num_items = len(self.clean_items)
        indices = np.arange(num_items)

        rng = np.random.RandomState(seed)
        rng.shuffle(indices)

        n_train = int(split_ratio[0] * num_items)
        n_val = int(split_ratio[1] * num_items)

        train_idx = indices[:n_train]
        val_idx = indices[n_train: n_train + n_val]
        test_idx = indices[n_train + n_val:]

        return train_idx, val_idx, test_idx

    def get_split(self, split):
        assert split in ['train', 'val', 'test', 'full']
        if split == 'train':
            return Subset(self, indices=self.training_indices)
        elif split == 'val':
            return Subset(self, indices=self.val_indices)
        elif split == 'test':
            return Subset(self, indices=self.test_indices)
        else:
            return self

    def __getitem__(self, idx):
        return {
            'item_id': self.clean_items[idx],
            'response': self.clean_response_matrix[:, idx],
            'mask': (self.clean_response_matrix[:, idx] == -1).astype(np.int8)
        }

    def __len__(self):
        return self.clean_response_matrix.shape[1]

    def normalize(self):
        matrix = self.clean_response_matrix

        row_missing = np.sum(matrix == -1, axis=1)
        col_missing = np.sum(matrix == -1, axis=0)
        total_missing = np.sum(matrix == -1)

        model_accuracy = (np.sum(matrix, axis=1) + row_missing) / (
            matrix.shape[1] - row_missing
        )
        item_accuracy = (np.sum(matrix, axis=0) + col_missing) / (
            matrix.shape[0] - col_missing
        )
        overall_acc = (np.sum(matrix) + total_missing) / (
            matrix.size - total_missing
        )

        self.clean_response_matrix = (
            matrix - model_accuracy[:, None] - item_accuracy[None, :] + overall_acc
        ).astype(np.float32)

    def prune(self, max_iter=10, drop_ratio=0.05):
        matrix = self.full_response_matrix
        missing = (matrix == -1)

        num_rows, num_cols = matrix.shape

        active_rows = np.ones(num_rows, dtype=bool)
        active_cols = np.ones(num_cols, dtype=bool)

        for it in range(max_iter):

            sub_missing = missing[np.ix_(active_rows, active_cols)]

            row_obs = 1 - sub_missing.mean(axis=1)
            col_obs = 1 - sub_missing.mean(axis=0)

            coverage = 1 - sub_missing.mean()

            logging.info(f"Pruning: Iter {it}: rows={active_rows.sum()}, cols={active_cols.sum()}, total_entries={active_rows.sum()*active_cols.sum()}, missing_entries={sub_missing.sum()}, coverage={coverage:.4f}")

            if coverage >= self.min_coverage:
                break

            rows_idx = np.where(active_rows)[0]
            cols_idx = np.where(active_cols)[0]

            k_row = max(1, int(len(row_obs) * drop_ratio))
            k_col = max(1, int(len(col_obs) * drop_ratio))

            drop_rows_local = np.argsort(row_obs)[:k_row]
            active_rows[rows_idx[drop_rows_local]] = False

            drop_cols_local = np.argsort(col_obs)[:k_col]
            active_cols[cols_idx[drop_cols_local]] = False

        sub_matrix = matrix[np.ix_(active_rows, active_cols)]

        drop_rows = np.where(~active_rows)[0].tolist()
        drop_columns = np.where(~active_cols)[0].tolist()

        return sub_matrix, drop_rows, drop_columns


if __name__ == '__main__':
    resp_dataset = ResponseDataset(input_file='data/response_matrix.csv')
