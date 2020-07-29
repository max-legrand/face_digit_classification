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


def create_zeros_array(size):
    array = []
    for counter in range(size):
        array.append(0)
    return array


def get_past_prob(global_obj: GlobalObj, data, training_size):
    for i in range(10):
        global_obj.count_one.append(0)
        global_obj.prev_prob.append(float(0))

    for i in range(training_size):
        global_obj.count_one[data[i]] += 1

    for i in range(10):
        global_obj.prev_prob[i] = float(global_obj.count_one[i] / training_size)


def get_prob(global_obj: GlobalObj, data_array, label_array, training_size):
    for i in range(10):
        global_obj.count_two.append(create_zeros_array(load_data.DIGIT_HEIGHT * load_data.DIGIT_WIDTH))
        global_obj.prob_array.append(create_zeros_array(load_data.DIGIT_HEIGHT * load_data.DIGIT_WIDTH))

    for i in range(training_size):
        feature = extract_feature(data_array[i])
        for j in range(len(feature)):
            if feature[j] == 1:
                global_obj.count_two[label_array[i]][j] += 1

    for i in range(load_data.DIGIT_HEIGHT * load_data.DIGIT_WIDTH):
        for j in range(10):
            global_obj.prob_array[j][i] = float((global_obj.count_two[j][i] + 0.01) / (global_obj.count_one[j] + 0.01))


def extract_feature(data):
    result = []
    for i in range(load_data.DIGIT_HEIGHT):
        for j in range(load_data.DIGIT_WIDTH):
            if len(data[i]) > 0 and data[i][j] != 0:
                result.append(1)
                continue
            result.append(0)
    return result


def naive_bayes_digit_train(global_obj: GlobalObj, training_size):
    get_past_prob(global_obj, global_obj.training_labels, training_size)
    get_prob(global_obj, global_obj.training_images, global_obj.training_labels, training_size)


def determine_digit(global_obj: GlobalObj, feat):
    miss = 0.0000000000001
    probability = []
    local_count = []
    for _ in range(10):
        local_count.append(0)
        probability.append(0)
    for count in range(len(feat)):
        if feat[count] == 1:
            for count_two in range(10):
                local_count[count_two] += log(global_obj.prob_array[count_two][count])
        else:
            for count_two in range(10):
                if(global_obj.prob_array[count_two][count] == 1):
                    global_obj.prob_array[count_two][count] -= miss
            for count_two in range(10):
                local_count[count_two] += log(1-global_obj.prob_array[count_two][count])

    for count in range(10):
        probability[count] = log(global_obj.prev_prob[count])+local_count[count]

    digits_results = {
                        '0': probability[0], '1': probability[1], '2': probability[2],
                        '3': probability[3], '4': probability[4], '5': probability[5],
                        '6': probability[6], '7': probability[7], '8': probability[8],
                        '9': probability[9],
                    }
    return max(digits_results, key=lambda item: digits_results[item])


def naive_bayes_digit_predict(global_obj: GlobalObj):
    results = []
    for item in range(len(global_obj.test_images)):
        value = determine_digit(global_obj, extract_feature(global_obj.test_images[item]))
        results.append(value)
    return results
