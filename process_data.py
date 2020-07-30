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


def create_zeros_array(size: int):
    """
    Creates an array with all 0s

    Args:
        size (int): size of array

    Returns:
        list: list containing all 0s
    """
    array = []
    for _ in range(size):
        array.append(0)
    return array


def get_past_prob_digit(global_obj: GlobalObj, data: list, training_size: int):
    """
    Determines the previous probablility and stroes to global object

    Args:
        global_obj (GlobalObj): GlobalObj to save data to
        data (list): labels data
        training_size (int): size of training data
    """
    for i in range(10):
        global_obj.digit_variables.count_one.append(0)
        global_obj.digit_variables.prev_prob.append(float(0))

    for i in range(training_size):
        global_obj.digit_variables.count_one[data[i]] += 1

    for i in range(10):
        global_obj.digit_variables.prev_prob[i] = float(global_obj.digit_variables.count_one[i] / training_size)


def get_probability_digit(global_obj: GlobalObj, data_array: list, label_array: list, training_size: int):
    """
    Determines current probability

    Args:
        global_obj (GlobalObj): GlobalObj to save data to
        data_array (list): images data array
        label_array (list): labels data array
        training_size (int): size of training data
    """
    for i in range(10):
        global_obj.digit_variables.count_two.append(
            create_zeros_array(load_data.DIGIT_HEIGHT * load_data.DIGIT_WIDTH)
        )
        global_obj.digit_variables.prob_array.append(create_zeros_array(
            load_data.DIGIT_HEIGHT * load_data.DIGIT_WIDTH)
        )

    for i in range(training_size):
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


def naive_bayes_digit_train(global_obj: GlobalObj, training_size: int):
    """
    Trains naive bayes model

    Args:
        global_obj (GlobalObj): GlobalObj to save data to
        training_size (int): size of training data
    """
    get_past_prob_digit(global_obj, global_obj.training_labels, training_size)
    get_probability_digit(global_obj, global_obj.training_images, global_obj.training_labels, training_size)


def determine_digit(global_obj: GlobalObj, feat: list):
    """
    Determines the most likely digit for an image

    Args:
        global_obj (GlobalObj): GlobalObj to save data to
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
        global_obj (GlobalObj): GlobalObj to save data to

    Returns:
        list: array of digits
    """
    results = []
    for item in range(len(global_obj.test_images)):
        value = determine_digit(global_obj, extract_feature(global_obj.test_images[item], True))
        results.append(value)
    return results


def naive_bayes_face_train(global_obj: GlobalObj, training_size: int):
    """
    Train the model using face data

    Args:
        global_obj (GlobalObj): GlobalObj to save data to
        training_size (int): size of training_data
    """
    count_faces(global_obj, global_obj.training_labels, training_size)
    get_past_prob_face(global_obj, training_size)
    get_probability_face(global_obj, training_size)


def get_past_prob_face(global_obj: GlobalObj, training_size: int):
    """
    Determines the previous probability based on face count

    Args:
        global_obj (GlobalObj): GlobalObj to save data to
        training_size (int): size of training data
    """
    global_obj.face_variables.prev_prob.append(
        float(
            global_obj.face_variables.not_a_face_count / training_size
        )
    )
    global_obj.face_variables.prev_prob.append(
        float(
            global_obj.face_variables.is_a_face_count / training_size
        )
    )


def count_faces(global_obj: GlobalObj, labels_array: list, training_size: int):
    """
    Counts number of faces present in the dataset

    Args:
        global_obj (GlobalObj): GlobalObj to save data to
        labels_array (list): labels array containing information about face
        training_size (int): size of training data
    """
    for i in range(0, training_size):
        if labels_array[i] == 0:
            global_obj.face_variables.not_a_face_count += 1
        else:
            global_obj.face_variables.is_a_face_count += 1


def get_probability_face(global_obj: GlobalObj, training_size: int):
    """
    Determines probablility that image is a face

    Args:
        global_obj (GlobalObj): GlobalObj to save data to
        training_size (int): size of training data
    """
    local_counts = []
    local_counts.append(global_obj.face_variables.not_a_face_count)
    local_counts.append(global_obj.face_variables.is_a_face_count)

    for _ in range(2):
        global_obj.face_variables.pixels_count.append(create_zeros_array(load_data.FACE_HEIGHT*load_data.FACE_WIDTH))
        global_obj.face_variables.prob_array.append(create_zeros_array(load_data.FACE_HEIGHT*load_data.FACE_WIDTH))

    for counter in range(training_size):
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
