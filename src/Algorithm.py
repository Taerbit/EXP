"""Class to deal with the """
from abc import ABC, abstractmethod

import shap
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from abc import ABC, abstractmethod
import keras.backend as K

class Algo(ABC):
    def pass_through(self, data, id):
        pass

    def get_input(self):
        pass


class empty(Algo):
    def __init__(self, name="None"):
        self.name = name
        return

    def pass_through(self, data, id):
        return

    def get_input(self):
        return ["Input_Image"]


class gradient_shap(Algo):

    name = "gradient_shap"

    def __init__(self, background, model, output_size, matrix_path="", img_path=""):

        self.e = shap.GradientExplainer(model, background)#np.array(background))
        self.img_path = img_path
        self.matrix_path = matrix_path
        self.image_counter = 0
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

    def visualization(self, matrix, image):
        shap.image_plot(np.array([matrix]), np.array([image]), show=False)
        plt.savefig(self.img_path + str(self.image_counter) + '.png')
        self.image_counter = self.image_counter + 1
        plt.clf()

    def pass_through(self, data, id):

        image = data[0]
        label = data[1]

        shap_value = self.e.shap_values(np.array([image]))

        # Reshape the shap value into the output size
        sorted_shap_value = self.shap_metric_ready(np.array(shap_value), label)

        if self.img_path != "":
            self.visualization(shap_value[label], image)

        if self.matrix_path != "":
            np.save(self.matrix_path + id + ".npy", np.array(shap_value))
            np.save(self.matrix_path + id + "_sorted.npy", sorted_shap_value)

        return sorted_shap_value

    def get_input(self):
        """Returns the names of the data types the algorithm needs"""
        return ["Input_Image", "Label"]

class grad_cam(Algo):

    name = "grad_cam"

    def __init__(self, model, output_size, layer_name, matrix_path="", img_path=""):
        self.model = self.create_gradcam_model(model, layer_name)
        self.output_size = output_size
        self.matrix_path = matrix_path
        self.img_path = img_path
        self.image_counter = 0
        self.matrix_counter = 0

    def create_gradcam_model(self, model, LAYER_NAMES):

        # Find all the layer outputs from the layers specified in the LAYER_NAMES list
        outputs = [
            layer.output for layer in model.layers
            if layer.name in LAYER_NAMES
        ]

        # Add the model's output layer to the list of outputs
        outputs.append(model.output)

        # Create and return the model
        grad_model = tf.keras.models.Model(model.inputs, outputs=outputs)
        return grad_model

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

    def visualization(self, cam, image, id):

        heatmap = (cam - cam.min()) / (cam.max() - cam.min())
        heatmap = cv2.resize(heatmap, (300, 225))

        heatmap = (heatmap * 255)
        plt.figure
        plt.imshow(image.astype('uint8'))
        plt.imshow((heatmap), cmap='jet', alpha=0.6)

        o = self.img_path + id + '.png'
        plt.savefig(o)

        plt.clf()

    def pass_through(self, data, id):
        img = data[0]
        label = data[1]
        original_image = data[2]

        # Get the score for target class
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.model(np.array([img]))
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
        cam = np.maximum(cam, 0)

        sorted_cam = cv2.resize(cam, (self.output_size[0], self.output_size[1]))
        #sorted_cam = np.maximum(cam, 0)  # ReLu

        if self.img_path != "":
            self.visualization(cam, original_image, id)

        if self.matrix_path != "":
            np.save(self.matrix_path + id + ".npy", cam)  # Save the raw cam output with dimensions of the last conv layer
            np.save(self.matrix_path + id + "_sorted.npy", sorted_cam) # Save

        return sorted_cam

    def get_input(self):
        """Returns the names of the data types the algorithm needs"""
        return ["Input_Image", "Label", "Original Image"]
"""

    Finn Torbet - 15/13/2020
    BSc Applied Computing Honours Project

"""