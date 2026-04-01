import sys
import os
import logging


local_path = {
    'raw_response_matrix': 'data/response_matrix.csv',
    'cached_response_matrix': 'data/response_matrix.pkl'
}


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
