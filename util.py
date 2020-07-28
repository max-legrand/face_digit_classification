#!/usr/bin/env pipenv run python
'''
file:           util.py
author:         Max Legrand
lastChangedBy:  Max Legrand
fileOverview:   File containing utility functions
'''


def convert_pixel(item):
    """
    Convert integer to pixel char and vise versa

    Args:
        item (int, char): integer / character that will be converted

    Returns:
        int, char: corresponding integer / character
    """
    if isinstance(item, int):
        data = {
            0: ' ',
            1: '#',
            2: '+'
        }
    else:
        data = {
            ' ': 0,
            '#': 1,
            '+': 2
        }
    return data[item]


def get_accuracy(results_data, label_data):
    """
    Function to determine the accuracy of a classification

    Args:
        results_data (list): results from running classification
        label_data (list): accurate raw data
    """
    percentage = float(0)
    for counter in range(len(results_data)):
        result = int(results_data[counter])
        raw = int(label_data[counter])
        if result == raw:
            percentage += 1
    percentage = percentage / len(results_data)
    return percentage
