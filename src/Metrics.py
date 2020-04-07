from abc import ABC, abstractmethod
import numpy as np

class Metric(ABC):

    name = "Metric"

    def process(self, matrix, segmentation):
        pass

    def validate_sizes(self, size1, size2):
        if size1.shape[:2] == size2.shape[:2]:
            return True
        else:
            print("ERROR: Incompatible sizes within metric processing: " +
                str(str(size1.shape) + " != " + str(size2.shape)))
            return False


    # Calculate two sets of activations from the grad_matrix, those inside the segmentation and those without
    def splitting_inside_and_outside(self, grad_matrix, segmentation, inside_colour):

        # Split the activations of the grad matrix to two sets, inside the segmentation and outside
        inside = []
        outside = []

        for h in range(len(segmentation)):
            for w in range(len(segmentation[h])):
                if segmentation[h][w][0] == inside_colour:
                    inside.append(grad_matrix[h][w])
                else:
                    outside.append(grad_matrix[h][w])

        return np.asarray(inside), np.asarray(outside)

    # Tagging each pixel wiht an outside or inside tag
    def tagging_inside_and_outside(self, grad_matrix, segmentation, inside_colour):

        dtype = [('Importance', np.float), ('Inside', np.bool)]
        g = np.zeros((len(grad_matrix) * len(grad_matrix[0])), dtype=dtype)
        x = []

        for h in range(len(segmentation)):
            for w in range(len(segmentation[h])):
                if segmentation[h][w][0] == inside_colour:
                    x.append(True)
                else:
                    x.append(False)

        g['Importance'] = grad_matrix.flatten()
        g['Inside'] = x

        return g



class Average(Metric):

    def __init__(self, inside_colour):
        self.inside_colour = inside_colour
        self.name = "Average"

    def process(self, matrix, segmentation):

        # Input validation of matching sizes
        if self.validate_sizes(matrix, segmentation):

            # Split the activations of the grad matrix to two sets, inside the segmentation and outside
            inside, outside = self.splitting_inside_and_outside(matrix, segmentation, self.inside_colour)

            i = self.__mean(inside)
            o = self.__mean(outside)

            return i/o

        else:

            return 0

    def __mean(self, list):
        n = len(list)
        sum = 0
        for x in list:
            sum += x
        return sum / n


class N(Metric):

    denominator_constant = 1

    def __init__(self, inside_colour, x):
        self.inside_colour = inside_colour
        self.x = x
        self.name = "N(" + str(x) + ")"

    def process(self, matrix, segmentation):

        # Input validation of matching sizes
        if self.validate_sizes(matrix, segmentation):

            # Split the activations of the grad matrix to two sets, inside the segmentation and outside
            g = self.tagging_inside_and_outside(matrix, segmentation, self.inside_colour)

            # Flatten and sort the grad-cam in descending order
            g = np.sort(g, order='Importance')
            g = np.flip(g)

            # Number of pixels inside the segmentation
            n_frg = int(np.sum(g['Inside']) / self.x)

            g = g[:n_frg]

            i_intersection = np.sum(g['Inside'])
            o_intersection = n_frg - i_intersection
            #print(f"{i_intersection + o_intersection} where n equals {n_frg}")

            output = i_intersection / (o_intersection + self.denominator_constant)

            return output

"""

    Finn Torbet - 15/13/2020
    BSc Applied Computing Honours Project

"""