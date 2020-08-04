#!/usr/bin/env pipenv run python
'''
file:           process_data.py
author:         Max Legrand
lastChangedBy:  Max Legrand
fileOverview:   File to test + train data
'''

from math import log
from global_obj import GlobalObj
import load_data
import numpy as np


def get_past_prob_digit(global_obj: GlobalObj, data: list, data_size: int):
    """
    Determines the previous probablility and stroes to global object

    Args:
        global_obj (GlobalObj): GlobalObj to save data to / read data from
        data (list): labels data
        data_size (int): size of training data
    """
    for i in range(10):
        global_obj.digit_variables.count_one.append(0)
        global_obj.digit_variables.prev_prob.append(float(0))

    for i in range(data_size):
        global_obj.digit_variables.count_one[data[i]] += 1

    for i in range(10):
        global_obj.digit_variables.prev_prob[i] = float(global_obj.digit_variables.count_one[i] / data_size)


def get_probability_digit(global_obj: GlobalObj, data_array: list, label_array: list, data_size: int):
    """
    Determines current probability

    Args:
        global_obj (GlobalObj): GlobalObj to save data to / read data from
        data_array (list): images data array
        label_array (list): labels data array
        data_size (int): size of training data
    """
    for i in range(10):
        global_obj.digit_variables.count_two.append(
            np.zeros(load_data.DIGIT_HEIGHT * load_data.DIGIT_WIDTH)
        )
        global_obj.digit_variables.prob_array.append(np.zeros(
            load_data.DIGIT_HEIGHT * load_data.DIGIT_WIDTH)
        )

    for i in range(data_size):
        feature = extract_feature(data_array[i], True)
        for j in range(len(feature)):
            if feature[j] == 1:
                global_obj.digit_variables.count_two[label_array[i]][j] += 1

    for i in range(load_data.DIGIT_HEIGHT * load_data.DIGIT_WIDTH):
        for j in range(10):
            global_obj.digit_variables.prob_array[j][i] = float(
                (
                    global_obj.digit_variables.count_two[j][i] + 0.01
                ) / (
                    global_obj.digit_variables.count_one[j] + 0.01
                )
            )


def extract_feature(data: list, is_digit: bool):
    """
    Extracts features from data

    Args:
        data (list): image array to extract features features from
        is_digit (bool): flag for if digit sizes should be used

    Returns:
        list: results of feature extraction
    """
    result = []

    if is_digit:
        height = load_data.DIGIT_HEIGHT
        width = load_data.DIGIT_WIDTH
    else:
        height = load_data.FACE_HEIGHT
        width = load_data.FACE_WIDTH
    for i in range(height):
        for j in range(width):
            if data[i][j] != 0:
                result.append(1)
                continue
            result.append(0)
    return result


def naive_bayes_digit_train(global_obj: GlobalObj, data_size: int):
    """
    Trains naive bayes model

    Args:
        global_obj (GlobalObj): GlobalObj to save data to / read data from
        data_size (int): size of training data
    """
    get_past_prob_digit(global_obj, global_obj.training_labels, data_size)
    get_probability_digit(global_obj, global_obj.training_images, global_obj.training_labels, data_size)


def determine_digit(global_obj: GlobalObj, feat: list):
    """
    Determines the most likely digit for an image

    Args:
        global_obj (GlobalObj): GlobalObj to save data to / read data from
        feat (list): features arrays

    Returns:
        string: most probable digit
    """
    miss = 0.0000000000001
    probability = []
    local_count = []
    for _ in range(10):
        local_count.append(0)
        probability.append(0)
    for count in range(len(feat)):
        if feat[count] == 1:
            for count_two in range(10):
                local_count[count_two] += log(global_obj.digit_variables.prob_array[count_two][count])
        else:
            for count_two in range(10):
                if(global_obj.digit_variables.prob_array[count_two][count] == 1):
                    global_obj.digit_variables.prob_array[count_two][count] -= miss
            for count_two in range(10):
                local_count[count_two] += log(1-global_obj.digit_variables.prob_array[count_two][count])

    for count in range(10):
        probability[count] = log(global_obj.digit_variables.prev_prob[count])+local_count[count]

    digits_results = {
                        '0': probability[0], '1': probability[1], '2': probability[2],
                        '3': probability[3], '4': probability[4], '5': probability[5],
                        '6': probability[6], '7': probability[7], '8': probability[8],
                        '9': probability[9],
                    }
    return max(digits_results, key=lambda item: digits_results[item])


def naive_bayes_digit_predict(global_obj: GlobalObj):
    """
    Predicts digits for test data

    Args:
        global_obj (GlobalObj): GlobalObj to save data to / read data from

    Returns:
        list: array of digits
    """
    results = []
    for item in range(len(global_obj.test_images)):
        value = determine_digit(global_obj, extract_feature(global_obj.test_images[item], True))
        results.append(value)
    return results


def naive_bayes_face_train(global_obj: GlobalObj, data_size: int):
    """
    Train the model using face data

    Args:
        global_obj (GlobalObj): GlobalObj to save data to / read data from
        data_size (int): size of training_data
    """
    count_faces(global_obj, global_obj.training_labels, data_size)
    get_past_prob_face(global_obj, data_size)
    get_probability_face(global_obj, data_size)


def get_past_prob_face(global_obj: GlobalObj, data_size: int):
    """
    Determines the previous probability based on face count

    Args:
        global_obj (GlobalObj): GlobalObj to save data to / read data from
        data_size (int): size of training data
    """
    global_obj.face_variables.prev_prob.append(
        float(
            global_obj.face_variables.not_a_face_count / data_size
        )
    )
    global_obj.face_variables.prev_prob.append(
        float(
            global_obj.face_variables.is_a_face_count / data_size
        )
    )


def count_faces(global_obj: GlobalObj, labels_array: list, data_size: int):
    """
    Counts number of faces present in the dataset

    Args:
        global_obj (GlobalObj): GlobalObj to save data to / read data from
        labels_array (list): labels array containing information about face
        data_size (int): size of training data
    """
    for i in range(0, data_size):
        if labels_array[i] == 0:
            global_obj.face_variables.not_a_face_count += 1
        else:
            global_obj.face_variables.is_a_face_count += 1


def get_probability_face(global_obj: GlobalObj, data_size: int):
    """
    Determines probablility that image is a face

    Args:
        global_obj (GlobalObj): GlobalObj to save data to / read data from
        data_size (int): size of training data
    """
    local_counts = []
    local_counts.append(global_obj.face_variables.not_a_face_count)
    local_counts.append(global_obj.face_variables.is_a_face_count)

    for _ in range(2):
        global_obj.face_variables.pixels_count.append(np.zeros(load_data.FACE_HEIGHT*load_data.FACE_WIDTH))
        global_obj.face_variables.prob_array.append(np.zeros(load_data.FACE_HEIGHT*load_data.FACE_WIDTH))

    for counter in range(data_size):
        features = extract_feature(global_obj.training_images[counter], False)
        for counter_two in range(len(features)):
            if features[counter_two] == 1:
                global_obj.face_variables.pixels_count[global_obj.training_labels[counter]][counter_two] += 1
    for counter in range(load_data.FACE_HEIGHT * load_data.FACE_WIDTH):
        for counter_two in range(2):
            global_obj.face_variables.prob_array[counter_two][counter] = float(
                (global_obj.face_variables.pixels_count[counter_two][counter] + 0.01) /
                (local_counts[counter_two] + 0.01)
            )


def determine_face(global_obj: GlobalObj, feat: list):
    """
    Determines the whether an image has a face or not

    Args:
        global_obj (GlobalObj): GlobalObj to save data to / read data from
        feat (list): features arrays

    Returns:
        string: "0" if no face, "1" if face present
    """
    miss = 0.0000000000001
    probability = []
    local_count = []
    for _ in range(2):
        local_count.append(0)
        probability.append(0)
    for count in range(len(feat)):
        if feat[count] == 1:
            for count_two in range(2):
                local_count[count_two] += log(global_obj.face_variables.prob_array[count_two][count])
        else:
            for count_two in range(2):
                if(global_obj.face_variables.prob_array[count_two][count] == 1):
                    global_obj.face_variables.prob_array[count_two][count] -= miss
            for count_two in range(2):
                local_count[count_two] += log(1-global_obj.face_variables.prob_array[count_two][count])

    for count in range(2):
        probability[count] = log(global_obj.face_variables.prev_prob[count])+local_count[count]

    face_results = {
                        '0': probability[0], '1': probability[1]
                    }
    return max(face_results, key=lambda item: face_results[item])


def naive_bayes_face_predict(global_obj: GlobalObj):
    """
    Predicts if image contains face

    Args:
        global_obj (GlobalObj): GlobalObj to save data to / read data from

    Returns:
        [list]: array containing whether each image is predicted to have a face or not
    """
    results = []
    for item in range(len(global_obj.test_images)):
        value = determine_face(global_obj, extract_feature(global_obj.test_images[item], False))
        results.append(value)
    return results


def perceptron_train_digit(global_obj: GlobalObj, data_size: int):
    """
    Train model using digit data

    Args:
        global_obj (GlobalObj): GlobalObj to read from / write to
        data_size (int): size of training data
    """
    get_weights(global_obj, data_size)


def get_weights(global_obj: GlobalObj, data_size: int):
    """
    Determines weights for each possible digit

    Args:
        global_obj (GlobalObj): GlobalObj to read data from / save data to
        data_size (int): size of training data
    """
    val_array = global_obj.digit_variables.val_array
    val_array_two = global_obj.digit_variables.val_array_two
    weights = global_obj.digit_variables.weights
    scores = global_obj.digit_variables.scores

    for _ in range(10):
        weights.append(np.zeros(784))

    for item in global_obj.digit_variables.weights:
        val_array.append(item)

    val_array_two = np.zeros(10)

    for counter in range(data_size):
        features = extract_feature(global_obj.training_images[counter], True)
        scores = []
        for counter_two in range(10):
            scores.append(
                np.dot(val_array[counter_two], features) +
                val_array_two[counter_two]
            )
        if global_obj.training_labels[counter] != scores.index(max(scores)):
            val_array[scores.index(max(scores))] = np.subtract(
                val_array[scores.index(max(scores))], np.asarray(features).transpose()
            )
            val_array[global_obj.training_labels[counter]] = np.add(
                val_array[global_obj.training_labels[counter]], np.asarray(features).transpose()
            )
            val_array_two[scores.index(max(scores))] = val_array_two[scores.index(max(scores))] - 1
            val_array_two[global_obj.training_labels[counter]] = val_array_two[global_obj.training_labels[counter]] + 1

    global_obj.digit_variables.val_array = val_array
    global_obj.digit_variables.val_array_two = val_array_two
    global_obj.digit_variables.weights = weights
    global_obj.digit_variables.scores = scores


def determine_digit_perceptron(global_obj: GlobalObj, feat: list):
    result = list(np.zeros(10, dtype=int))
    val_array = global_obj.digit_variables.val_array
    val_array_two = global_obj.digit_variables.val_array_two
    for counter in range(10):
        result[counter] = np.dot(val_array[counter], feat) + val_array_two[counter]
    digit_results = {
        "0": result[0],
        "1": result[1],
        "2": result[2],
        "3": result[3],
        "4": result[4],
        "5": result[5],
        "6": result[6],
        "7": result[7],
        "8": result[8],
        "9": result[9],
    }
    value = max(digit_results, key=lambda item: digit_results[item])
    return int(value)


def perceptron_digit_predict(global_obj: GlobalObj):
    results = []
    for counter in range(len(global_obj.test_images)):
        results.append(determine_digit_perceptron(global_obj, extract_feature(global_obj.test_images[counter], True)))
    return results
