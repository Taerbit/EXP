import EXP
import numpy as np
import tensorflow as tf
import cv2

def set_up():
    # Set up for the test object
    model = tf.keras.models.load_model("test_data\\200310_DenseNet201_001.h5", compile=False)
    all_imgs = EXP.Input.get_all_paths("test_data\\Lesions\\Lesions\\", "*.jpg")
    background = []
    for i in range(len(all_imgs)):
        background.append(np.array([EXP.Input.load_input_image(all_imgs[i], (255, 300))]))

    return EXP.Algorithm.gradient_shap(background, model, [1022, 767])

def test_gradient_shap():
    """Expecting input in the form of the set up section
    Then expecting output similar to that of the pre-created data files"""

    # Construct the test object
    gs = set_up()

    # Load in the data to test the object on
    data = [
            EXP.Input.load_input_image("test_data\\Lesions\\Lesions\\ISIC_0000000.jpg", (225, 300)),
            0
        ]
    print(data[0])
    x = np.load("test_data\\shap_0_sorted.npy")

    sorted_gs = gs.pass_through(data)
    np.save("test.npy", sorted_gs)
    assert sorted_gs == x

def test_shap_metric_ready():
    unsorted = np.load("test_data\\shap_1.npy")
    sorted = np.load("test_data\\shap_1_sorted.npy")

    gs = set_up()

    x_sorted = gs.shap_metric_ready(unsorted, 0)
    assert np.array_equal(sorted, x_sorted)

def test_grad_cam():
    model = tf.keras.models.load_model("test_data\\200310_DenseNet201_001.h5", compile=False)
    gradcam = EXP.Algorithm.grad_cam(model, [1022, 767])

    assert gradcam.get_input() == ["Input_Image", "Label"]

    img = EXP.Input.load_input_image("test_data\\Lesions\\Lesions\\ISIC_0000000.jpg", (225, 300))

    x = np.load("test_data\\gradcam_0.npy")

    g = gradcam.pass_through([img, 0])
    assert g == x

test_gradient_shap()