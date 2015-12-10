#!/usr/bin/env python
import sys
import argparse
import tdt

DEFAULT_ITERATION = 10000

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a tensordot code by random search.")
    tdt.add_default_arguments(parser)
    parser.add_argument('iteration', nargs='?',
                        type=int, default=DEFAULT_ITERATION,
                        help='the number of random search try (default: {})'.format(DEFAULT_ITERATION))
    args = parser.parse_args()

    tdt.main(args,rand_flag=True)
