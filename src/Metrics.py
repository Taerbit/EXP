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

    # Splits the matrix into a set of inside and outside activations through matrix multiplicaiton
    def splitting(self, m, s, inside_colour):

        # Drop all but one axis in the binary segmentation mask
        s = np.delete(s, 0, 2)
        s = np.delete(s, 0, 2)
        s = s.squeeze()

        #Normalize the binary segmentation
        s = s / inside_colour

        # Times the matrix by the normalized segmentation to get only the inside values
        inside = m * s

        # Flip the segmentation and do the same to get outside
        s = 1-s
        outside = m*s

        # Strip all inside and outside values
        inside = inside[inside != 0]
        outside = outside[outside != 0]

        return inside, outside



class Average(Metric):

    def __init__(self, inside_colour):
        self.inside_colour = inside_colour
        self.name = "Average"

    def process(self, matrix, segmentation):

        # Input validation of matching sizes
        if self.validate_sizes(matrix, segmentation):

            # Split the activations of the grad matrix to two sets, inside the segmentation and outside
            inside, outside = self.splitting(matrix, segmentation, self.inside_colour)

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

    def __init__(self, inside_colour, x):
        self.inside_colour = inside_colour
        self.x = x
        self.name = "N(" + str(x) + ")"

    def process(self, matrix, segmentation):

        # Input validation of matching sizes
        if self.validate_sizes(matrix, segmentation):

            # Retreive the two sets
            inside, outside = self.splitting(matrix, segmentation, self.inside_colour)

            # Number of pixels inside the segmentation
            n_frg = int( len(inside)/self.x )

            # Flattening the grad-matrix and sorting gives the sorted set of Inside and Outside together
            # which can be used to find the minimum. The min can be used to cut inside and outside sets
            sorted_matrix = matrix.flatten()
            sorted_matrix = np.sort(sorted_matrix)
            sorted_matrix = sorted_matrix[::-1]

            min = sorted_matrix[n_frg]
            inside = inside[inside >= min]
            outside = outside[outside >= min]

            output = len(inside) / (len(inside)+len(outside))

            return output

"""

    Finn Torbet - 15/13/2020
    BSc Applied Computing Honours Project

"""