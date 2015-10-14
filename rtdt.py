#!/usr/bin/env python
import sys
import tdt

if __name__ == "__main__":
    if len(sys.argv)<2:
        sys.exit("Usage: ./tdtr.py input_file [iteration]")

    iteration = 0 if len(sys.argv)<3 else int(sys.argv[2])
    tdt.main(True,iteration)
