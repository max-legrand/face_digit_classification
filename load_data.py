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
test = []
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

    # Read in images file
    images_contents = [line.strip() for line in open(images_filename).readlines()]
    shuffled_images_contents = []
    # Read in labels file
    labels_content = [line.strip() for line in open(label_filename).readlines()]
    shuffled_labels_content = []

    # use appropriate height
    if isdigit:
        height = DIGIT_HEIGHT
    else:
        height = FACE_HEIGHT

    for counter in range(datasize):
        index = random.randint(0, len(labels_content)-1)
        shuffled_labels_content.append(labels_content.pop(index))

        for image_counter in range(index*height, index*height + height):
            shuffled_images_contents.append(images_contents.pop(image_counter))

    labels_results = []
    images_results = []

    for counter in range(datasize):
        labels_results.append(int(shuffled_labels_content[counter]))
        temp_array = []
        for image_counter in range(height):
            row = shuffled_images_contents.pop()
            convert_row = []
            for item in row:
                convert_row.append(convert_pixel(item))
            temp_array.append(convert_row)
        images_results.append(temp_array)
    return images_results, labels_results


# Test functions and verify proper operation
if __name__ == "__main__":
    print(convert_pixel(1))
    print(convert_pixel('#'))
    images, labels = load_file_contents("data/digitdata/testimages", 1, "data/digitdata/testlabels", True)
    print(labels)
    print(images)
