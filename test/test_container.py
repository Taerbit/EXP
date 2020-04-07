import EXP
import cv2
import numpy as np

output_size = (128, 128)
input_size = (255, 300)


def check_images_match(p1, p2):
    if p1.shape == p2.shape:
        difference = cv2.subtract(p1, p2)
        b, g, r = cv2.split(difference)
        if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
            return True
    return False


def test_segmentation_base_functions():
    s = s = EXP.Input.Segmentation(["test_data\\", "test.png", 0],
                               output_size)

    assert s.get_number() == 1
    assert s.increment()
    assert s.get_number() == 2
    s.increment()
    assert s.increment() == False

def test_segmentation_loading():
    s = EXP.Input.Segmentation(["test_data\\", "test.png", 0],
                               output_size)

    # Test Loading
    filepaths = ["C:\\Users\\finnt\PycharmProjects\\EXP\\test\\test_data\\0001test.png",
                 "C:\\Users\\finnt\PycharmProjects\\EXP\\test\\test_data\\0002test.png",
                 "C:\\Users\\finnt\PycharmProjects\\EXP\\test\\test_data\\0003test.png"]
    img = []
    for i in range(3):
        image = cv2.imread(filepaths[i])
        img.append(cv2.resize(image, output_size))

    assert check_images_match(s.load(), img[0])
    assert check_images_match(s.load(), img[1])
    assert check_images_match(s.load(), img[2])

def test_input_image():
    i = EXP.Input.Input_Image(["test_data\\", ".jpg"], input_size)

    img = i.load()

    x = EXP.Input.load_input_image("test_data\\input0.jpg", input_size)

    assert np.array_equals(img, x)

test_input_image()


'''

    self.assertTrue(s.get_number(), 2, msg="First number is not returned correctly")
    s.increment()
    self.assertFalse(s.increment(), 1, msg="Correct Truth value is not returned by the increment whilst out of the list")


self.assertTrue(s.increment())#,
                    #msg="Correct Truth value is not returned by the increment whilst still in the list")
    self.assertTrue(s.get_number())#, 2, msg="First number is not returned correctly")
    s.increment()
    self.assertFalse(s.increment())#, 1,
                     #msg="Correct Truth value is not returned by the increment whilst out of the list")


'''
