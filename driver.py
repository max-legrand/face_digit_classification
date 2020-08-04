#!/usr/bin/env pipenv run python
'''
file:           driver.py
author:         Max Legrand
lastChangedBy:  Max Legrand
fileOverview:   Driver to run classifications
'''

import argparse
import timeit
from statistics import mean, stdev
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
from util import get_accuracy


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

    args = parser.parse_args()

    if args.bayes:
        if args.digit:
            format_print("Naive Bayes Digit Classification:")
            # Iterate over training sizes in increments of 10%
            for counter in range(0, 10):
                results = []
                time_results = []

                # Do 5 iterations of training + testing
                for iteration in range(0, 5):
                    print(f"Run #{iteration+1 + 5*counter}")
                    training_size = 500*(counter+1)
                    GLOBALS.load_digit_data(training_size)

                    print(f"Using training size: {training_size}")
                    start_time = timeit.default_timer()

                    # TRAIN THE DATA
                    naive_bayes_digit_train(GLOBALS, training_size)

                    end_time = timeit.default_timer()
                    delta_time = end_time - start_time
                    print("Training Finished...")
                    print(f"Elapsed Time: {delta_time} sec")
                    time_results.append(delta_time)
                    print("Testing Data")
                    # TEST THE DATA
                    prediction_results = naive_bayes_digit_predict(GLOBALS)
                    results.append(get_accuracy(prediction_results, GLOBALS.test_labels))
                    print(f"Accuracy: {results[iteration]}%")
                format_print(
                                "Summary\n" + f"Training Size: {10*(counter+1)}%\n" +
                                f"Average Training Time: {mean(time_results)} sec\n" +
                                f"Mean Accuracy: {mean(results)}%\n" +
                                f"Standard Deviation: {stdev(results)}%\n"
                            )

        elif args.face:
            format_print("Naive Bayes Faces Classification:")
            # Iterate over training sizes in increments of 10%
            for counter in range(0, 10):
                results = []
                time_results = []

                # Do 5 iterations of training + testing
                for iteration in range(0, 5):
                    print(f"Run #{iteration+1 + 5*counter}")
                    training_size = 45*(counter+1)
                    GLOBALS.load_face_data(training_size)

                    print(f"Using training size: {training_size}")
                    start_time = timeit.default_timer()

                    # TRAIN THE DATA
                    naive_bayes_face_train(GLOBALS, training_size)

                    end_time = timeit.default_timer()
                    delta_time = end_time - start_time
                    print("Training Finished...")
                    print(f"Elapsed Time: {delta_time} sec")
                    time_results.append(delta_time)
                    print("Testing Data")
                    # TEST THE DATA
                    prediction_results = naive_bayes_face_predict(GLOBALS)
                    results.append(get_accuracy(prediction_results, GLOBALS.test_labels))
                    print(f"Accuracy: {results[iteration]}%")
                format_print(
                                "Summary\n" + f"Training Size: {10*(counter+1)}%\n" +
                                f"Average Training Time: {mean(time_results)} sec\n" +
                                f"Mean Accuracy: {mean(results)}%\n" +
                                f"Standard Deviation: {stdev(results)}%\n"
                            )

    elif args.percep:
        if args.digit:
            format_print("Perceptron Digit Classification:")
            # Iterate over training sizes in increments of 10%
            for counter in range(0, 10):
                results = []
                time_results = []

                # Do 5 iterations of training + testing
                for iteration in range(0, 5):
                    print(f"Run #{iteration+1 + 5*counter}")
                    training_size = 500*(counter+1)
                    GLOBALS.load_digit_data(training_size)

                    print(f"Using training size: {training_size}")
                    start_time = timeit.default_timer()

                    # TRAIN THE DATA
                    perceptron_train_digit(GLOBALS, training_size)

                    end_time = timeit.default_timer()
                    delta_time = end_time - start_time
                    print("Training Finished...")
                    print(f"Elapsed Time: {delta_time} sec")
                    time_results.append(delta_time)
                    print("Testing Data")
                    # TEST THE DATA
                    prediction_results = perceptron_digit_predict(GLOBALS)
                    results.append(get_accuracy(prediction_results, GLOBALS.test_labels))
                    print(f"Accuracy: {results[iteration]}%")
                format_print(
                                "Summary\n" + f"Training Size: {10*(counter+1)}%\n" +
                                f"Average Training Time: {mean(time_results)} sec\n" +
                                f"Mean Accuracy: {mean(results)}%\n" +
                                f"Standard Deviation: {stdev(results)}%\n"
                            )

        elif args.face:
            format_print("Perceptron Face Classification:")
            # Iterate over training sizes in increments of 10%
            for counter in range(0, 10):
                results = []
                time_results = []

                # Do 5 iterations of training + testing
                for iteration in range(0, 5):
                    print(f"Run #{iteration+1 + 5*counter}")
                    training_size = 45*(counter+1)
                    GLOBALS.load_face_data(training_size)

                    print(f"Using training size: {training_size}")
                    start_time = timeit.default_timer()

                    # TRAIN THE DATA
                    perceptron_train_face(GLOBALS, training_size)

                    end_time = timeit.default_timer()
                    delta_time = end_time - start_time
                    print("Training Finished...")
                    print(f"Elapsed Time: {delta_time} sec")
                    time_results.append(delta_time)
                    print("Testing Data")
                    # TEST THE DATA
                    prediction_results = perceptron_face_predict(GLOBALS)
                    results.append(get_accuracy(prediction_results, GLOBALS.test_labels))
                    print(f"Accuracy: {results[iteration]}%")
                format_print(
                                "Summary\n" + f"Training Size: {10*(counter+1)}%\n" +
                                f"Average Training Time: {mean(time_results)} sec\n" +
                                f"Mean Accuracy: {mean(results)}%\n" +
                                f"Standard Deviation: {stdev(results)}%\n"
                            )
