import numpy as np
import Input
import shap
import matplotlib.pyplot as plt

img = Input.load_reg_image("..\\test\\test_data\\Lesions\\97\\ISIC_0000000.jpg", 300, 255)
img = img.squeeze()
shap_v = np.load("C:\\Users\\finnt\\Documents\\Honours Results\\200416_DenseNet201_001\\shap\\ISIC_0000021.npy")

o = shap.image_plot(shap_v, img)
plt.show()