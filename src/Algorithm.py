"""Class to deal with the """
from abc import ABC, abstractmethod

import shap
import numpy as np
import matplotlib.pyplot as plt
import cv2

class empty():
    def __init__(self):
        return

    def pass_through(self, image, label):
        return label


class gradient_shap():

    def __init__(self, background, model, output_size, matrix_path="", img_path=""):
        self.e = shap.GradientExplainer(model, np.array(background))
        self.model = model
        self.img_path = img_path
        self.matrix_path = matrix_path
        self.image_counter = 0
        self.matrix_counter = 0
        self.output_size = output_size

    def shap_metric_ready(self, shap, label):
        shap = shap.squeeze()
        shap = shap[label]

        metric_ready = np.zeros(shap.shape)
        metric_ready = metric_ready[:, :, 0]

        for y, h in enumerate(shap):
            for x, w in enumerate(h):
                total = sum(w)
                metric_ready[y][x] = total

        metric_ready = cv2.resize(metric_ready, (self.output_size[0], self.output_size[1]))

        return metric_ready

    def pass_through(self, image, label):

        shap_value = self.e.shap_values(image)

        shap_value = self.shap_metric_ready(shap_value, label)

        if self.img_path != "":
            shap.image_plot(np.array([shap_value]), -np.array([image]), show=False)
            plt.savefig(self.img_path + str(self.image_counter) + '.png')
            plt.clf()

        if self.matrix_path != "":
            np.save(self.matrix_path + str(self.matrix_counter) + ".npy", shap_value)

        return shap_value

"""

    Finn Torbet - 15/13/2020
    BSc Applied Computing Honours Project

"""