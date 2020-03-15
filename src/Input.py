from abc import ABC, abstractmethod
import cv2
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import pandas as pd


"""

    Input classes to take in data
        Constructor data
            Containers come as a list of container inheritted classes
        Expected Methods:
            - has_next() : Returns a boolean for if there is more data to come
            - get_next() : Returns next frame of data
            - get_id() : Return an identification of the current dataframe

"""

class Sorter:
    """Given a list of filenames, find the numerical value for each filepath and make sure they match up"""

    def __init__(self, containters):
        self.containers = containters

    def validate(self, a):
        """Checks if all zero or positive values match"""
        unique_value = a[np.where(a > -1)]
        for x in a:
            if x == -1:
                continue
            if x != unique_value:
                return False
        return True

    def find_smallest(self, a):
        min = a[np.where(a > -1)]

    def has_next(self):
        for c in self.containers:
            if not c.data_remaining() == True:
                return False
        return True

    def get_next(self):
        """Returns the next lot of data in the order of containers"""
        while True:
            check = np.empty(len(self.containers))
            for i, c in enumerate(self.containers):
                check[i] = c.get_number()

            if self.validate(check) == True:

                loaded_data = []
                for l in self.containers:
                    loaded_data.append(l.load())
                return loaded_data

            else:

                smallest = self.find_smallest(check)
                if self.containers[smallest].increment() == False:
                    return False

    def get_id(self):
        # Sort through data for best fit (a flag in a container [set at construction] to indicate a 'identification' container )
        return self.containers[0].get_number() # Just returns the first container's current name


class Linear_Loader():

    def __init__(self, container):
        self.c = container

    def has_next(self):
        return self.c.data_remaining()

    def get_next(self):
        return self.c.load()


"""

    Container Classes to hold data to be fed into a system
        Constructor data:
            Given filepaths for loading and sorting the data
            Given tags to strip filepaths down to key identifiers (which can be used by the Input classes to sort)
            Given ordered to inidcate if a certain element is to be missed in any ordering
        Expected Methods:
            - load() : Will return the data
            
"""


class Container(ABC):

    def __init__(self, filepaths, tags, ordered):
        self.counter = 0
        self.fp = filepaths  # List of filepaths in order
        self.tags = tags     # List of elements of the filepaths that are not
        self.ordered = ordered


    @abstractmethod
    def load(self):
        """Load the current file at position counter in the filepaths array"""
        pass

    def get_number(self):
        """Return the number from the current filepath at counter"""
        if self.ordered == True:
            strip = self.fp[self.counter]

            for tag in self.tags:
                if tag == int:
                    strip.lstrip(str(tag))
                else:
                    strip = strip.replace(str(tag), '')

            return int(strip)
        else:
            return -1

    def data_remaining(self):
        """Increment the counter by one"""
        self.counter = self.counter + 1
        if self.counter+1 == len(self.fp):
            return False
        else:
            return True

class Segmentation(Container):

    def __init__(self, filepaths, tags, output_size, ordered=True):
        super(Segmentation, self).__init__(filepaths, tags, ordered)
        self.output_size = output_size

    def load(self):
        image = cv2.imread(self.fp[self.counter])
        image = cv2.resize(image, self.output_size)
        self.counter = self.counter + 1
        return image, self.data_remaining()

class Input_Image(Container):

    def __init__(self, filepaths, tags, input_size, ordered=True):
        super(Input_Image, self).__init__(filepaths, tags, ordered)
        self.input_size = input_size

    def load(self):
        image = load_img(self.fp[self.counter], target_size=(225, 300))
        image = img_to_array(image)

        def normalize(x):
            return (x / 128) - 1

        image = normalize(image)
        self.counter = self.counter + 1
        return image, self.data_remaining()

class Matrix(Container):
    """Load in metric ready, precompiled matrices from a previous algorithms generation"""
    def __init__(self, filepaths, tags, ordered=True):
        super(Matrix, self).__init__(filepaths, tags, ordered)

    def load(self):
        matrix = np.load(self.fp[self.counter])
        self.counter = self.counter + 1
        return matrix, self.data_remaining()

class Label(Container):

    def __init__(self, filepath, column_header, tags=[""], ordered=True):
        super(Label, self).__init__(filepath, tags, ordered)

        # Load information now
        data = pd.read_csv(filepath)
        self.fp = data[column_header]

    def load(self):
        self.counter = self.counter + 1
        return self.fp[self.counter], self.data_remaining()


    def get_number(self):
        return self.fp[self.counter]
