import argparse
import torch
import json
import logging
import os
import pickle
from src.data import ResponseDataset
from pathlib import Path
from src.utils import setup_logger, str2bool
from src.config import path
from src.sae_extract import LassoSparseAutoencoder, TopKSparseAutoEncoder
from src.trainer import SAETrainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model setting
    parser.add_argument('--model_type', type=str, default='lasso')  # lasso, topk
    parser.add_argument('--hidden_dim', type=int)
    parser.add_argument('--l1_weight', type=float)
    parser.add_argument('--tie_weights', type=bool, default=True)
    parser.add_argument('--min_activation', type=float)
    parser.add_argument('--aux_ratio', type=float)
    parser.add_argument('--dead_steps', type=int)

    # training setting
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--verbose', type=str2bool, default=True)
    parser.add_argument('--use_full', type=str2bool, default=True)
    parser.add_argument('--do_val', type=str2bool, )
    parser.add_argument('--do_test', type=str2bool, )
    parser.add_argument('--do_inference', type=str2bool, default=True)
    parser.add_argument('--early_stop', type=str2bool, default=True)
    # io
    parser.add_argument('--model_save_dir', type=Path, )
    parser.add_argument('--model_load_dir', type=Path)
    parser.add_argument('--log_file', type=Path, default=None)
    parser.add_argument('--feature_save_dir', type=Path)

    args = parser.parse_args()

    setup_logger(log_file=args.log_file)
    env = os.environ.get("CUR_ENV")
    # dataset
    if os.path.exists(path[env]['cached_response_matrix']):
        with open(path[env]['cached_response_matrix'], 'rb') as f_data:
            response_dataset = pickle.load(f_data)
    else:
        response_dataset = ResponseDataset(path[env]['raw_response_matrix'], min_coverage=0.8)
        with open(path[env]['cached_response_matrix'], "wb") as f_data:
            pickle.dump(response_dataset, f_data)

    logging.info(f'{len(response_dataset.clean_items)} items, {len(response_dataset.clean_subjects)} subjects')

    # model
    model = LassoSparseAutoencoder(input_dim=len(response_dataset.clean_subjects), hidden_dim=args.hidden_dim)

    # train
    trainer = SAETrainer(
        sae_model=model,
        dataset=response_dataset,
        l1_weight=args.l1_weight,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=torch.device('cuda' if torch.cuda.is_available() else'cpu'),
        do_val=args.do_val,
        do_test=args.do_test,
        min_activation=args.min_activation,
        early_stop=args.early_stop,
        dead_steps=args.dead_steps,
        aux_weight=args.l1_weight * 5,
        auk_ratio=args.aux_ratio,
    )
    trainer.train(use_full=args.use_full)
    trainer.load_best()

    if args.do_test:
        trainer.eval(split='test')
        logging.info(f"Test recon loss: {trainer.test_tracker.get_last()['loss_recon']:.2f}"
                     f" | Avg active: {trainer.test_tracker.get_last()['avg_active']:.2f}"
                     f" | Total active: {trainer.test_tracker.get_last()['total_active']}/{trainer.model.hidden_dim}")

    model_id = f"S{len(response_dataset.clean_subjects)}_I{len(response_dataset.clean_items)}_N{args.hidden_dim}_L{args.l1_weight}_E{trainer.best_epoch}"

    # save generated features
    if args.do_inference:
        feature_save_path = os.path.join(args.feature_save_dir, f"features_{model_id}.json")
        output = trainer.eval(split='full')
        with open(feature_save_path, 'w') as fp_features:
            json.dump(output, fp_features)

    # save model
    if args.model_save_dir:
        model_save_path = os.path.join(args.model_save_dir, f'SAE_{model_id}.pt')
        trainer.save(save_path=model_save_path, save_best=True)


