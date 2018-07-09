import unittest
from argparse import ArgumentParser
import logging

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--verbose', '-v', action='count', default=0,
                        help='Logging verbosity: repeat for more output')
    parser.add_argument('--output', '-o',
                        help='Path to log file (default stderr)')
    parsed_args = parser.parse_args()
    log_level = {
        0: logging.CRITICAL,
        1: logging.CRITICAL,
        2: logging.WARNING,
        3: logging.INFO,
        4: logging.DEBUG
    }.get(parsed_args.verbose, logging.DEBUG)

    kwargs = {'level': log_level}

    if parsed_args.output:
        kwargs['filename'] = parsed_args.output
        kwargs['filemode'] = 'w'

    logging.basicConfig(**kwargs)
    from .cases import *  # to log any import-related issues

    logging.getLogger().setLevel(log_level)

    unittest.main(verbosity=2 if parsed_args.verbose else 1)
