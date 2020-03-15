from . import Input as I
import cv2
import matplotlib.pyplot as plt


def check_images_match(p1, p2):
    if p1.shape == p2.shape:
        difference = cv2.subtract(p1, p2)
        b, g, r = cv2.split(difference)
        if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
            return True
    return False


def test_segmentation_base_functions():
    filepaths = ["C:\\Users\\finnt\\PycharmProjects\\EXP\\src\\test_data\\hello world 00001.png",
                 "C:\\Users\\finnt\\PycharmProjects\\EXP\\src\\test_data\\hello world 00002.png",
                 "C:\\Users\\finnt\\PycharmProjects\\EXP\\src\\test_data\\hello world 00003.png"]
    output_size = (128, 128)

    s = I.Segmentation(filepaths,
                       ["C:\\Users\\finnt\\PycharmProjects\\EXP\\src\\test_data\\hello world ", 0, ".png"],
                       output_size)

    x = s
    assert s.get_number() == 1
    assert s.increment() == True
    assert s.get_number() == 2
    s.increment()
    assert s.increment() == False

    # Test Loading
    img = []
    for i in range(3):
        image = cv2.imread(filepaths[i])
        #plt.imshow(image)
        #img.append(cv2.resize(image, (output_size[0], output_size[1])))

    assert check_images_match(x.load(), img[0]) == True
    assert check_images_match(x.load(), img[1]) == True
    assert check_images_match(x.load(), img[2]) == True

test_segmentation_base_functions()
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
