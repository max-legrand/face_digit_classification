#!/usr/bin/env pipenv run python
'''
file:           process_data.py
author:         Max Legrand
lastChangedBy:  Max Legrand
fileOverview:   File to test + train data
'''

from math import log
import numpy as np
import load_data
from global_obj import GlobalObj


def get_past_prob_digit(globject: GlobalObj, data: list, data_size: int):
    """
    Determines the previous probablility and stroes to global object

    Args:
        globject (GlobalObj): GlobalObj to save data to / read data from
        data (list): labels data
        data_size (int): size of training data
    """
    for i in range(10):
        globject.digit_variables_bayes.count_one.append(0)
        globject.digit_variables_bayes.prev_prob.append(float(0))

    for i in range(data_size):
        globject.digit_variables_bayes.count_one[data[i]] += 1

    for i in range(10):
        globject.digit_variables_bayes.prev_prob[i] = float(globject.digit_variables_bayes.count_one[i] / data_size)


def get_probability_digit(globject: GlobalObj, data_array: list, label_array: list, data_size: int):
    """
    Determines current probability

    Args:
        globject (GlobalObj): GlobalObj to save data to / read data from
        data_array (list): images data array
        label_array (list): labels data array
        data_size (int): size of training data
    """
    for i in range(10):
        globject.digit_variables_bayes.count_two.append(
            np.zeros(load_data.DIGIT_HEIGHT * load_data.DIGIT_WIDTH)
        )
        globject.digit_variables_bayes.prob_array.append(np.zeros(
            load_data.DIGIT_HEIGHT * load_data.DIGIT_WIDTH)
        )

    for i in range(data_size):
        feature = extract_feature(data_array[i], True)
        for j in range(len(feature)):
            if feature[j] == 1:
                globject.digit_variables_bayes.count_two[label_array[i]][j] += 1

    for i in range(load_data.DIGIT_HEIGHT * load_data.DIGIT_WIDTH):
        for j in range(10):
            globject.digit_variables_bayes.prob_array[j][i] = float(
                (
                    globject.digit_variables_bayes.count_two[j][i] + 0.01
                ) / (
                    globject.digit_variables_bayes.count_one[j] + 0.01
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


def naive_bayes_digit_train(globject: GlobalObj, data_size: int):
    """
    Trains naive bayes model

    Args:
        globject (GlobalObj): GlobalObj to save data to / read data from
        data_size (int): size of training data
    """
    get_past_prob_digit(globject, globject.training_labels, data_size)
    get_probability_digit(globject, globject.training_images, globject.training_labels, data_size)


def determine_digit(globject: GlobalObj, feat: list):
    """
    Determines the most likely digit for an image

    Args:
        globject (GlobalObj): GlobalObj to save data to / read data from
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
                local_count[count_two] += log(globject.digit_variables_bayes.prob_array[count_two][count])
        else:
            for count_two in range(10):
                if(globject.digit_variables_bayes.prob_array[count_two][count] == 1):
                    globject.digit_variables_bayes.prob_array[count_two][count] -= miss
            for count_two in range(10):
                local_count[count_two] += log(1-globject.digit_variables_bayes.prob_array[count_two][count])

    for count in range(10):
        probability[count] = log(globject.digit_variables_bayes.prev_prob[count])+local_count[count]

    digits_results = {
                        '0': probability[0], '1': probability[1], '2': probability[2],
                        '3': probability[3], '4': probability[4], '5': probability[5],
                        '6': probability[6], '7': probability[7], '8': probability[8],
                        '9': probability[9],
                    }
    return max(digits_results, key=lambda item: digits_results[item])


def naive_bayes_digit_predict(globject: GlobalObj):
    """
    Predicts digits for test data

    Args:
        globject (GlobalObj): GlobalObj to save data to / read data from

    Returns:
        list: array of digits
    """
    results = []
    for item in range(len(globject.test_images)):
        value = determine_digit(globject, extract_feature(globject.test_images[item], True))
        results.append(value)
    return results


def naive_bayes_face_train(globject: GlobalObj, data_size: int):
    """
    Train the model using face data

    Args:
        globject (GlobalObj): GlobalObj to save data to / read data from
        data_size (int): size of training_data
    """
    count_faces(globject, globject.training_labels, data_size)
    get_past_prob_face(globject, data_size)
    get_probability_face(globject, data_size)


def get_past_prob_face(globject: GlobalObj, data_size: int):
    """
    Determines the previous probability based on face count

    Args:
        globject (GlobalObj): GlobalObj to save data to / read data from
        data_size (int): size of training data
    """
    globject.face_variables_bayes.prev_prob.append(
        float(
            globject.face_variables_bayes.not_a_face_count / data_size
        )
    )
    globject.face_variables_bayes.prev_prob.append(
        float(
            globject.face_variables_bayes.is_a_face_count / data_size
        )
    )


def count_faces(globject: GlobalObj, labels_array: list, data_size: int):
    """
    Counts number of faces present in the dataset

    Args:
        globject (GlobalObj): GlobalObj to save data to / read data from
        labels_array (list): labels array containing information about face
        data_size (int): size of training data
    """
    for i in range(0, data_size):
        if labels_array[i] == 0:
            globject.face_variables_bayes.not_a_face_count += 1
        else:
            globject.face_variables_bayes.is_a_face_count += 1


def get_probability_face(globject: GlobalObj, data_size: int):
    """
    Determines probablility that image is a face

    Args:
        globject (GlobalObj): GlobalObj to save data to / read data from
        data_size (int): size of training data
    """
    local_counts = []
    local_counts.append(globject.face_variables_bayes.not_a_face_count)
    local_counts.append(globject.face_variables_bayes.is_a_face_count)

    for _ in range(2):
        globject.face_variables_bayes.pixels_count.append(np.zeros(load_data.FACE_HEIGHT*load_data.FACE_WIDTH))
        globject.face_variables_bayes.prob_array.append(np.zeros(load_data.FACE_HEIGHT*load_data.FACE_WIDTH))

    for counter in range(data_size):
        features = extract_feature(globject.training_images[counter], False)
        for counter_two in range(len(features)):
            if features[counter_two] == 1:
                globject.face_variables_bayes.pixels_count[globject.training_labels[counter]][counter_two] += 1
    for counter in range(load_data.FACE_HEIGHT * load_data.FACE_WIDTH):
        for counter_two in range(2):
            globject.face_variables_bayes.prob_array[counter_two][counter] = float(
                (globject.face_variables_bayes.pixels_count[counter_two][counter] + 0.01) /
                (local_counts[counter_two] + 0.01)
            )


def determine_face(globject: GlobalObj, feat: list):
    """
    Determines the whether an image has a face or not

    Args:
        globject (GlobalObj): GlobalObj to save data to / read data from
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
                local_count[count_two] += log(globject.face_variables_bayes.prob_array[count_two][count])
        else:
            for count_two in range(2):
                if(globject.face_variables_bayes.prob_array[count_two][count] == 1):
                    globject.face_variables_bayes.prob_array[count_two][count] -= miss
            for count_two in range(2):
                local_count[count_two] += log(1-globject.face_variables_bayes.prob_array[count_two][count])

    for count in range(2):
        probability[count] = log(globject.face_variables_bayes.prev_prob[count])+local_count[count]

    face_results = {
                        '0': probability[0], '1': probability[1]
                    }
    return max(face_results, key=lambda item: face_results[item])


def naive_bayes_face_predict(globject: GlobalObj):
    """
    Predicts if image contains face

    Args:
        globject (GlobalObj): GlobalObj to save data to / read data from

    Returns:
        [list]: array containing whether each image is predicted to have a face or not
    """
    results = []
    for item in range(len(globject.test_images)):
        value = determine_face(globject, extract_feature(globject.test_images[item], False))
        results.append(value)
    return results


def perceptron_train_digit(globject: GlobalObj, data_size: int):
    """
    Train model using digit data

    Args:
        globject (GlobalObj): GlobalObj to read from / write to
        data_size (int): size of training data
    """
    get_weights_digits(globject, data_size)


def get_weights_digits(globject: GlobalObj, data_size: int):
    """
    Determines weights for each possible digit

    Args:
        globject (GlobalObj): GlobalObj to read data from / save data to
        data_size (int): size of training data
    """
    val_array = globject.percep_variables.val_array
    val_array_two = globject.percep_variables.val_array_two
    weights = globject.percep_variables.weights
    scores = globject.percep_variables.scores

    for _ in range(10):
        weights.append(np.zeros(784))

    for item in globject.percep_variables.weights:
        val_array.append(item)

    val_array_two = np.zeros(10)

    for counter in range(data_size):
        features = extract_feature(globject.training_images[counter], True)
        scores = []
        for counter_two in range(10):
            scores.append(
                np.dot(val_array[counter_two], features) +
                val_array_two[counter_two]
            )
        if globject.training_labels[counter] != scores.index(max(scores)):
            val_array[scores.index(max(scores))] = np.subtract(
                val_array[scores.index(max(scores))], np.asarray(features).transpose()
            )
            val_array[globject.training_labels[counter]] = np.add(
                val_array[globject.training_labels[counter]], np.asarray(features).transpose()
            )
            val_array_two[scores.index(max(scores))] = val_array_two[scores.index(max(scores))] - 1
            val_array_two[globject.training_labels[counter]] = val_array_two[globject.training_labels[counter]] + 1

    globject.percep_variables.val_array = val_array
    globject.percep_variables.val_array_two = val_array_two
    globject.percep_variables.weights = weights
    globject.percep_variables.scores = scores


def determine_digit_perceptron(globject: GlobalObj, feat: list):
    """
    Determines which digit the image best is represented by

    Args:
        globject (GlobalObj): GlobalObj to read data from / write data to
        feat (list): features array

    Returns:
        [int]: most likely digit
    """
    result = list(np.zeros(10, dtype=int))
    val_array = globject.percep_variables.val_array
    val_array_two = globject.percep_variables.val_array_two
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
    return max(digit_results, key=lambda item: digit_results[item])


def perceptron_digit_predict(globject: GlobalObj):
    """
    Predicts most likely digit

    Args:
        globject (GlobalObj): GlobalObj to read data from / write data to

    Returns:
        list: results for each image
    """
    results = []
    for counter in range(len(globject.test_images)):
        results.append(determine_digit_perceptron(globject, extract_feature(globject.test_images[counter], True)))
    return results


def perceptron_train_face(globject: GlobalObj, data_size: int):
    """
    Train perceptron model using face data

    Args:
        globject (GlobalObj): GlobalObj to read data from / write data to
        data_size (int): training data size
    """

    get_weights_face(globject, data_size)


def get_weights_face(globject: GlobalObj, data_size: int):
    """
    Determine the weight for if each image is a face or not

    Args:
        globject (GlobalObj): GlobalObj to read data from / write data to
        data_size (int): training data size
    """
    val_array = globject.percep_variables.val_array
    val_array_two = globject.percep_variables.val_array_two
    weights = globject.percep_variables.weights
    scores = globject.percep_variables.scores

    for _ in range(2):
        weights.append(np.zeros(4200))

    for item in weights:
        val_array.append(item)

    val_array_two = np.zeros(2)

    for counter in range(data_size):
        feats = extract_feature(globject.training_images[counter], False)
        scores = []
        for counter_two in range(2):
            scores.append(
                np.dot(val_array[counter_two], feats) +
                val_array_two[counter_two]
            )

        if globject.training_labels[counter] != scores.index(max(scores)):
            val_array[scores.index(max(scores))] = np.subtract(
                val_array[scores.index(max(scores))], np.asarray(feats).transpose()
            )
            val_array[globject.training_labels[counter]] = np.add(
                val_array[globject.training_labels[counter]], np.asarray(feats).transpose()
            )
            val_array_two[scores.index(max(scores))] = val_array_two[scores.index(max(scores))] - 1
            val_array_two[globject.training_labels[counter]] = val_array_two[globject.training_labels[counter]] + 1

    globject.percep_variables.val_array = val_array
    globject.percep_variables.val_array_two = val_array_two
    globject.percep_variables.weights = weights
    globject.percep_variables.scores = scores


def perceptron_face_predict(globject: GlobalObj):
    """
    Predicts if each image is a face or not

    Args:
        globject (GlobalObj): GlobalObj to read data from / write data to

    Returns:
        list: array containing results for each image
    """
    results = []
    for counter in range(len(globject.test_images)):
        results.append(determine_face_perceptron(globject, extract_feature(globject.test_images[counter], False)))
    return results


def determine_face_perceptron(globject: GlobalObj, feats: list):
    """
    Determines whether each image is most likely a face or not

    Args:
        globject (GlobalObj): GlobalObj to read data from / write data to
        feats (list): features array

    Returns:
        string: string containing 0 if image isn't a face; 1 otherwise
    """
    result = list(np.zeros(2, dtype=int))
    val_array = globject.percep_variables.val_array
    val_array_two = globject.percep_variables.val_array_two
    for counter in range(2):
        result[counter] = np.dot(val_array[counter], feats) + val_array_two[counter]
    face_results = {
        "0": result[0],
        "1": result[1],
    }
    return max(face_results, key=lambda item: face_results[item])
