#!/usr/bin/env pipenv run python
'''
file:           single_test.py
author:         Max Legrand
lastChangedBy:  Max Legrand
fileOverview:   File to perform testing / training and display single image
'''

import argparse
from global_obj import GlobalObj
from process_data import (
    naive_bayes_digit_train,
    naive_bayes_digit_predict,
    naive_bayes_face_predict,
    naive_bayes_face_train,
    perceptron_train_digit,
    perceptron_digit_predict,
    perceptron_train_face,
    perceptron_face_predict,
)
from load_data import pretty_print_single


GLOBALS = GlobalObj()


def format_print(text):
    """
    Prints text inside banner

    Args:
        text (string): text to format
    """
    print("=====================================")
    print(text)
    print("=====================================")


if __name__ == "__main__":

    # Get arguments via command line input
    parser = argparse.ArgumentParser(description='Face and Digit Detection')
    datatype = parser.add_mutually_exclusive_group(required=True)
    datatype.add_argument('--digit', action="store_true")
    datatype.add_argument('--face', action="store_true")
    classtype = parser.add_mutually_exclusive_group(required=True)
    classtype.add_argument('--bayes', action="store_true")
    classtype.add_argument('--percep', action="store_true")
    parser.add_argument("--image", required=True)

    args = parser.parse_args()
    img = int(args.image)

    if args.digit and (img < 0 or img > 999):
        print("Invalid Image")
        exit()

    if args.face and (img < 0 or img > 149):
        print("Invalid Image")
        exit()

    if args.bayes:
        if args.digit:

            format_print("Naive Bayes Digit:")

            training_size = 5000
            GLOBALS.load_digit_data(training_size)
            print("Training Data")
            # TRAIN THE DATA
            naive_bayes_digit_train(GLOBALS, training_size)

            print("Testing Data")
            # TEST THE DATA
            prediction_results = naive_bayes_digit_predict(GLOBALS)
            pretty_print_single(GLOBALS.test_images[img])
            format_print(f"Predicted: {prediction_results[img]}\nActual {GLOBALS.test_labels[img]}")

        elif args.face:
            format_print("Naive Bayes Face:")

            training_size = 450
            GLOBALS.load_face_data(training_size)
            print("Training Data")
            # TRAIN THE DATA
            naive_bayes_face_train(GLOBALS, training_size)

            print("Testing Data")
            # TEST THE DATA
            prediction_results = naive_bayes_face_predict(GLOBALS)
            pretty_print_single(GLOBALS.test_images[img])
            format_print(f"Predicted: {prediction_results[img]}\nActual {GLOBALS.test_labels[img]}")

    elif args.percep:
        if args.digit:
            format_print("Perceptron Digit:")

            training_size = 5000
            GLOBALS.load_digit_data(training_size)
            print("Training Data")
            # TRAIN THE DATA
            perceptron_train_digit(GLOBALS, training_size)

            print("Testing Data")
            # TEST THE DATA
            prediction_results = perceptron_digit_predict(GLOBALS)
            pretty_print_single(GLOBALS.test_images[img])
            format_print(f"Predicted: {prediction_results[img]}\nActual {GLOBALS.test_labels[img]}")

        elif args.face:
            format_print("Perceptron Face:")

            training_size = 450
            GLOBALS.load_face_data(training_size)
            print("Training Data")
            # TRAIN THE DATA
            perceptron_train_face(GLOBALS, training_size)

            print("Testing Data")
            # TEST THE DATA
            prediction_results = perceptron_face_predict(GLOBALS)
            pretty_print_single(GLOBALS.test_images[img])
            format_print(f"Predicted: {prediction_results[img]}\nActual {GLOBALS.test_labels[img]}")
