import unittest
from argparse import ArgumentParser
import logging

logging.basicConfig(level=logging.WARNING)  # to pick up import-related logs
logger = logging.getLogger(__name__)

from cases import *

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--verbosity', '-v', action='count', default=0)
    parsed_args = parser.parse_args()
    log_level = {
        0: logging.CRITICAL,
        1: logging.WARNING,
        2: logging.INFO,
        3: logging.DEBUG
    }.get(parsed_args.verbosity, logging.DEBUG)
    logging.getLogger().setLevel(log_level)
    logging.getLogger('tensorflow').setLevel(log_level)

    unittest.main(verbosity=2 if parsed_args.verbosity else 1)
