import unittest
import tensorflow as tf
import Algorithm


class Testing(unittest.TestCase):

    def setUp(self):
        model = tf.keras.applications.vgg16.VGG16(weights='imagenet')
        algo = Algorithm.gradient_shap(0, model, 0)

    def test_construction(self):
        with self.assertRaises(AssertionError):
            x = Algorithm.gradient_shap(0, 0, 0)

    def test_explainer_construction(self):
        self.assertEquals



if __name__ == 'main':
    unittest.main()