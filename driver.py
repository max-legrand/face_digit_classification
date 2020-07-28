#!/usr/bin/env pipenv run python
'''
file:           driver.py
author:         Max Legrand
lastChangedBy:  Max Legrand
fileOverview:   Driver to run classifications
'''

import argparse
from global_obj import GlobalObj

GLOBALS = GlobalObj()


if __name__ == "__main__":

    # Get arguments via command line input
    parser = argparse.ArgumentParser(description='Face and Digit Detection')
    datatype = parser.add_mutually_exclusive_group(required=True)
    datatype.add_argument('--digit', action="store_true")
    datatype.add_argument('--face', action="store_true")
    classtype = parser.add_mutually_exclusive_group(required=True)
    classtype.add_argument('--bayes', action="store_true")
    classtype.add_argument('--percep', action="store_true")

    args = parser.parse_args()

    if args.bayes:
        if args.digit:
            print("Naive Bayes Digit Classification:")
            # Iterate over training sizes in increments of 10%
            for counter in range(0, 10):
                results = []
                time_results = []

                # Do 10 iterations of training + testing
                for iteration in range(0, 10):
                    print(f"Iteration: {iteration+1 + 10*counter}")
                    training_size = 500*(counter+1)
                    GLOBALS.load_digit_data(training_size)

        elif args.face:
            pass

    elif args.percep:
        if args.digit:
            pass
        elif args.face:
            pass
