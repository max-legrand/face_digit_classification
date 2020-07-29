#!/usr/bin/env pipenv run python
'''
file:           global_obj.py
author:         Max Legrand
lastChangedBy:  Max Legrand
fileOverview:   Class file for global data
'''

import load_data


class GlobalObj:
    def __init__(self):
        super().__init__()
        self.training_images = []
        self.training_labels = []
        self.verify_images = []
        self.verify_labels = []
        self.test_images = []
        self.test_labels = []
        self.prev_prob = []
        self.prob_array = []
        self.count_one = []
        self.count_two = []

    def load_digit_data(self, data_size):
        self.training_images, self.training_labels = load_data.load_file_contents(
            "data/digitdata/trainingimages",
            data_size,
            "data/digitdata/traininglabels",
            True)
        self.verify_images, self.verify_labels = load_data.load_file_contents(
            "data/digitdata/validationimages",
            1000,
            "data/digitdata/validationlabels",
            True)
        self.test_images, self.test_labels = load_data.load_file_contents(
            "data/digitdata/testimages",
            1000,
            "data/digitdata/testlabels",
            True)

    def load_face_data(self, data_size):
        self.training_images, self.training_labels = load_data.load_file_contents(
            "data/facedata/facedatatrain",
            data_size,
            "data/facedata/facedatatrainlabels",
            True)
        self.verify_images, self.verify_labels = load_data.load_file_contents(
            "data/facedata/facedatavalidation",
            301,
            "data/facedata/facedatavalidationlabels",
            True)
        self.test_images, self.test_labels = load_data.load_file_contents(
            "data/facedata/facedatatest",
            150,
            "data/facedata/facedatatestlabels",
            True)
