#!/usr/bin/env pipenv run python
'''
file:           global_obj.py
author:         Max Legrand
lastChangedBy:  Max Legrand
fileOverview:   Class file for global data
'''

from prodict import Prodict
import load_data


class GlobalObj:  # pylint: disable=too-many-instance-attributes
    def __init__(self):
        """
        Constructor for GlobalObj class
        """
        super().__init__()
        self.training_images = []
        self.training_labels = []
        self.verify_images = []
        self.verify_labels = []
        self.test_images = []
        self.test_labels = []
        self.digit_variables_bayes = Prodict.from_dict({
            "prev_prob": [],
            "prob_array": [],
            "count_one": [],
            "count_two": [],
        })
        self.face_variables_bayes = Prodict.from_dict({
            "not_a_face_count": 0,
            "is_a_face_count": 0,
            "prev_prob": [],
            "prob_array": [],
            "pixels_count": [],
        })
        self.percep_variables = Prodict.from_dict({
            "weights": [],
            "weights_array": [],
            "bias_array": [],
            "scores": []
        })

    def load_digit_data(self, data_size):
        """
        Loads data for digits

        Args:
            data_size (int): size of training data
        """
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
        """
        Loads data for faces

        Args:
            data_size (int): size of training data
        """
        self.training_images, self.training_labels = load_data.load_file_contents(
            "data/facedata/facedatatrain",
            data_size,
            "data/facedata/facedatatrainlabels",
            False)
        self.verify_images, self.verify_labels = load_data.load_file_contents(
            "data/facedata/facedatavalidation",
            301,
            "data/facedata/facedatavalidationlabels",
            False)
        self.test_images, self.test_labels = load_data.load_file_contents(
            "data/facedata/facedatatest",
            150,
            "data/facedata/facedatatestlabels",
            False)
