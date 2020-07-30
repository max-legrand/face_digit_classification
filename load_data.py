#!/usr/bin/env pipenv run python
'''
file:           load_data.py
author:         Max Legrand
lastChangedBy:  Max Legrand
fileOverview:   File containing functions to load data
'''
import random
from util import convert_pixel

# Globals to use with other files
DIGIT_WIDTH = 28
DIGIT_HEIGHT = 28
FACE_WIDTH = 60
FACE_HEIGHT = 70


def load_file_contents(images_filename, datasize, label_filename, isdigit):
    """
    Function which loads image and label data and returns arrays containing appropriate
    amount of shuffled data

    Args:
        images_filename (string): path for images data
        datasize (int): number of items to return
        label_filename (string): path for labels data
        isdigit (bool): flag for if data is digits. True if digits, false if faces

    Returns:
        [list, list]: returns images list and labels list
    """

    if isdigit:
        height = DIGIT_HEIGHT
    else:
        height = FACE_HEIGHT

    image_data = [line[:-1] for line in open(images_filename).readlines()]
    label_data = [line[:-1] for line in open(label_filename).readlines()]
    images = []

    total_size = len(label_data)
    rand_order = []

    for counter in range(total_size):
        rand_order.append(counter)

    random.shuffle(rand_order)
    shuffled_data = []

    for index in rand_order:
        for counter in range(height):
            shuffled_data.append(image_data[index*height + counter])

    shuffled_data.reverse()

    for _ in range(datasize):
        temp = []
        for _ in range(height):
            val = shuffled_data.pop()
            row_data = []
            for item in val:
                row_data.append(convert_pixel(item))
            temp.append(row_data)
        images.append(temp)

    shuffled_data = []

    for index in rand_order:
        shuffled_data.append(label_data[index])

    labels = []
    for counter in range(datasize):
        labels.append(int(shuffled_data[counter]))

    return images, labels


def pretty_print(images):
    """
    Prints image as plain text

    Args:
        images (list): array containing images in array format
    """
    for image in images:
        for row in image:
            for item in row:
                print(item, end="")
            print("")


# Test functions and verify proper operation
if __name__ == "__main__":
    # TEST CONVERT_PIXEL FUNCTION
    # print(convert_pixel(1))
    # print(convert_pixel('#'))

    # LOAD DIGIT DATA TEST
    # images_arr, labels_arr = load_file_contents("data/digitdata/testimages", 3, "data/digitdata/testlabels", True)
    # print(labels_arr)
    # pretty_print(images_arr)

    # LOAD FACE DATA TEST
    images_arr, labels_arr = load_file_contents(
        "data/facedata/facedatatest",
        3,
        "data/facedata/facedatatestlabels",
        False
    )
    print(len(images_arr[0]))
    print(labels_arr)
    pretty_print(images_arr)
