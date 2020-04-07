"""Class to deal with the """
from abc import ABC, abstractmethod

import shap
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from abc import ABC, abstractmethod

class Algo(ABC):
    def pass_through(self, data):
        pass

    def get_input(self):
        pass


class empty(Algo):
    def __init__(self, name="None"):
        self.name = name
        return

    def pass_through(self, data):
        return data[0]

    def get_input(self):
        return ["Matrix"]


class gradient_shap(Algo):

    name = "shap"

    def __init__(self, background, model, output_size, matrix_path="", img_path=""):
        self.e = shap.GradientExplainer(model, np.array(background))
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
                total = sum(abs(w))
                metric_ready[y][x] = total

        metric_ready = cv2.resize(metric_ready, (self.output_size[0], self.output_size[1]))

        return metric_ready

    def pass_through(self, data):

        image = np.array([data[0]])
        label = data[1]

        print(image)
        shap_value = self.e.shap_values(image)

        shap_value = self.shap_metric_ready(shap_value, label)

        if self.img_path != "":
            shap.image_plot(np.array([shap_value]), -np.array([image]), show=False)
            plt.savefig(self.img_path + str(self.image_counter) + '.png')
            self.img_path = self.img_path + 1
            plt.clf()

        if self.matrix_path != "":
            np.save(self.matrix_path + str(self.matrix_counter) + ".npy", shap_value)
            self.matrix_counter = self.matrix_counter + 1

        return shap_value

    def get_input(self):
        """Returns the names of the data types the algorithm needs"""
        return ["Input_Image", "Label"]

class grad_cam(Algo):

    def __init__(self, model, output_size, matrix_path="", img_path=""):
        self.model = model
        self.output_size = output_size
        self.matrix_path = matrix_path
        self.img_path = img_path
        self.image_counter = 0
        self.matrix_counter = 0

    # method to return a grad-cam score that has been unaltered for a given image and class on a model
    def normalise(self, grad_matrix):
        return (grad_matrix - grad_matrix.min()) / (grad_matrix.max() - grad_matrix.min())

    def validate_sizes(self, size1, size2):
        if size1.shape[:2] == size2.shape[:2]:
            return True
        else:
            print("ERROR: Incompatible sizes within Grad-CAM processing: " +
                str(str(size1.shape) + " != " + str(size2.shape)))
            return False

    def grad_cam_coloured(self, grad_matrix, img):

        img = cv2.resize(img, (len(grad_matrix[0]), len(grad_matrix)))
        output_image = 0

        if self.validate_sizes(grad_matrix, img):
            cam = cv2.applyColorMap(np.uint8(255 * self.normalise(grad_matrix)), cv2.COLORMAP_JET)
            output_image = cv2.addWeighted(cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR), 0.5,
                                           cam.astype('uint8'), 1, 0)

        return output_image

    def pass_through(self, data):
        image = data[0]
        label = data[1]

        # Get the score for target class
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.model(image)
            loss = predictions[:, label]

        # Extract filters and gradients
        output = conv_outputs[0]
        grads = tape.gradient(loss, conv_outputs)[0]

        # Average gradients spatially
        weights = tf.reduce_mean(grads, axis=(0, 1))

        # Build a ponderated map of filters according to gradients importance
        cam = np.ones(output.shape[0:2], dtype=np.float32)

        for index, w in enumerate(weights):
            cam += w * output[:, :, index]

        # Heatmap visualization
        cam = cv2.resize(cam.numpy(), (self.output_size[0], self.output_size[1]))
        cam = np.maximum(cam, 0)  # ReLu

        if self.img_path != "":
            overlapped = self.grad_cam_coloured(cam, image)
            plt.imshow(overlapped, show=False)
            plt.savefig(self.img_path + str(self.image_counter) + '.png')
            self.img_path = self.img_path + 1
            plt.clf()

        if self.matrix_path != "":
            np.save(self.matrix_path + str(self.matrix_counter) + ".npy", cam)
            self.matrix_counter = self.matrix_counter + 1

        return cam

    def get_input(self):
        """Returns the names of the data types the algorithm needs"""
        return ["Input_Image", "Label"]
"""

    Finn Torbet - 15/13/2020
    BSc Applied Computing Honours Project

"""