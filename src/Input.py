import re
from abc import ABC, abstractmethod
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import pathlib
import natsort
from abc import ABC, abstractmethod
from keras.preprocessing import image

"""

    Input_Loader classes to take in data
        Constructor data
            Containers come as a list of container inheritted classes
        Expected Methods:
            - has_next() : Returns a boolean for if there is more data to come
            - get_next() : Returns next frame of data
            - get_id() : Return an identification of the current dataframe

"""

class Input_Loader(ABC):
    def has_next(self):
        pass

    def get_next(self):
        pass

    def get_id(self):
        pass

class Sorter(Input_Loader):
    """Given a list of filenames, find the numerical value for each filepath and make sure they match up"""

    def __init__(self, containers, id_index=0):
        self.containers = containers
        self.id_index = id_index

    def load_frame(self):
        load = {}
        for c in self.containers:
            load[c.name] = c.load()
        return load

    def has_next(self):
        for c in self.containers:
            if not c.data_remaining():
                return False
        return True


    def match(self, a):
        max = np.amax(a)
        for i, n in enumerate(a):
            if n != -1 and max != n:
                return i
        return -1

    def get_all_numbers(self):
        numbers = []
        for c in self.containers:
            numbers.append(c.get_number())
        return numbers

    def get_next(self):
        """Returns the next lot of data in the order of containers"""

        while True:

            match = self.match(self.get_all_numbers())

            if match == -1:
                id = self.containers[self.id_index].get_id()
                return self.load_frame(), id
            else:
                self.containers[match].increment()


class Linear_Loader(Input_Loader):
    """Class for loading only one data type"""

    def __init__(self, container):
        self.c = container

    def has_next(self):
        return self.c.data_remaining()

    def get_next(self):
        return self.c.load()

    def get_id(self):
        return self.c.get_number()


"""

    Container Classes to hold data to be fed into a system
        Constructor data:
            Given filepaths for loading and sorting the data
            Given tags to strip filepaths down to key identifiers (which can be used by the Input classes to sort)
            Given ordered to inidcate if a certain element is to be missed in any ordering
        Expected Methods:
            - load() : Will return the data
            
"""

def get_all_paths(src, file_extension):
    image_root = pathlib.Path(src)
    all_paths = list(image_root.glob(file_extension))
    all_paths = [str(path) for path in all_paths]
    all_paths = natsort.natsorted(all_paths)
    return all_paths

#IMPROVEMENT    Make fp a stack or queue to remove counter variable

class Container(ABC):

    def __init__(self, tags, ordered, children):
        self.counter = 0
        self.fp = get_all_paths(tags[0], "*" + tags[1])  # List of filepaths in order
        self.tags = tags     # List of elements of the filepaths that are not
        self.ordered = ordered
        self.children = children


    @abstractmethod
    def load(self):
        """Load the current file at position counter in the filepaths array"""
        pass

    def increment(self):
        self.counter = self.counter + 1
        if not self.children == []:
            for c in self.children:
                if not c.data_remaining():
                    return False
                else:
                    c.increment()
        return self.data_remaining()

    def get_number(self):
        """Return the number from the current filepath at counter"""
        if self.ordered:
            strip = self.fp[self.counter]

            for tag in self.tags:
                if isinstance(tag, int):
                    strip.lstrip(str(tag))
                    if strip == "":
                        strip = str(tag)
                else:
                    strip = strip.replace(str(tag), '')

            return int(strip)
        else:
            return -1

    def data_remaining(self):
        """getter method asking if there is any data remaining.
        Return True if there is data still to be retreived,
        False if not"""

        if self.counter == len(self.fp):
            return False
        else:
            return True


    def get_id(self):
        id = self.fp[self.counter]
        id = id.replace(str(self.tags[0]), "")
        id = id.replace(str(self.tags[1]), "")
        return id



class Segmentation(Container):

    def __init__(self, tags, output_size, ordered=True, children=[]):
        super(Segmentation, self).__init__(tags, ordered, children)
        self.output_size = output_size
        self.name = "Segmentation"

    def load(self):
        image = cv2.imread(self.fp[self.counter])
        image = cv2.resize(image, self.output_size)
        self.counter = self.counter + 1
        return image

'''
def load_input_image(src, target_size):
    image = load_img(src, target_size=target_size)
    image = img_to_array(image)

    def normalize(x):
        return (x / 128) - 1

    image = normalize(image)
    return image
'''

def load_input_image(src, x, y):
    image = tf.io.read_file(src)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.dtypes.cast(image, tf.float32)
    image /= 127.5 - 1  # normalize to [-1,1]
    image = tf.image.resize_with_pad(image, y, x)

    return image

class Input_Image(Container):

    def __init__(self, tags, size, ordered=True, children=[]):
        super(Input_Image, self).__init__(tags, ordered, children)
        self.size = size
        self.name= "Input_Image"

    def load(self):
        image = load_input_image(self.fp[self.counter], self.size[0], self.size[1])
        #image = np.expand_dims(image, 0)
        self.counter = self.counter + 1
        return image



class Original_Image(Container):

    def __init__(self, tags, size, ordered=True, children=[]):
        super(Original_Image, self).__init__(tags, ordered, children)
        self.size = size
        self.name= "Original Image"

    def load(self):
        i = image.load_img(self.fp[self.counter], target_size=(self.size[1], self.size[0]))
        i = image.img_to_array(i)
        self.counter = self.counter + 1
        return i

class Matrix(Container):
    """Load in metric ready, precompiled matrices from a previous algorithms generation"""
    def __init__(self, tags, ordered=True, children=[]):
        super(Matrix, self).__init__(tags, ordered, children)
        self.name = "Matrix"

    def load(self):
        matrix = np.load(self.fp[self.counter])
        self.counter = self.counter + 1
        return matrix

class Label(Container):
    """Container class to read a csv for a label
        fp is the row ID
        labels is the actual label value"""

    def __init__(self, tag, data_header, name_header, ordered=True, children=[]):

        super(Label, self).__init__(tag, ordered, children)

        # Load information from csv
        data = pd.read_csv(self.fp[0])
        self.fp = data[name_header]
        self.fp = self.fp.values.tolist()

        if type(data_header) == list:
            self.multi = True
        else:
            self.multi = False

        self.labels = data[data_header]
        self.labels = self.labels.values.tolist()

        self.name = "Label"

    def load(self):
        if self.multi:
            l = np.argmax(self.labels[self.counter])
        else:
            l = self.labels[self.counter]

        self.counter = self.counter + 1
        return l

    def sort(self):
        self.fp = self.fp.sort

"""

    Finn Torbet - 15/13/2020
    BSc Applied Computing Honours Project

"""
