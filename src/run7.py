from EXP import Controller
from tensorflow.keras.models import load_model
import efficientnet.tfkeras
import tensorflow as tf

model= [
        #tf.keras.models.load_model("..\\models\\200310_DenseNet201_001.h5", compile=False),
        "..\\models\\200416_DenseNet201_001.h5",
        "..\\models\\200324_EfficientNetB0NoisyStudent_001.h5",
        "..\\models\\200411_EfficientNetB7NoisyStudent_001.h5"
]

output = [
    #"C:\\Users\\finnt\\Documents\\Honours Results\\200310_DenseNet201_001\\",
    "C:\\Users\\finnt\\Documents\\Honours Results\\200416_DenseNet201_001\\",
    "C:\\Users\\finnt\\Documents\\Honours Results\\200324_EfficientNetB0NoisyStudent_001\\",
    "C:\\Users\\finnt\\Documents\\Honours Results\\200411_EfficientNetB7NoisyStudent_001\\",
]

image_tags = ["..\\imgs\\Lesions\\all\\", ".jpg", "ISIC_", "_downsampled", 0]
seg_tags = ["C:\\Users\\finnt\\OneDrive\\Documents\\Uni\\Year 4\\Honours\\Project\\Segmentations\\", ".png", "ISIC_", "_segmentation", 0]
label_tag = ["..\\models\\", ".csv", "ISIC_", "_downsampled", 0]

input_tags = [image_tags, seg_tags, label_tag]

for i in range(len(model)):
    m = load_model(model[i])
    Controller.hons(m, input_tags, ["conv5_block32_2_conv"], output=output[i], save_csv=True, save_matrices=True, save_imgs=True)
    m.layers[-1].activation=tf.keras.activations.linear
    output[i] = output[i] + "Linear\\"
    Controller.hons(m, input_tags, ["conv5_block32_2_conv"], output=output[i], save_csv=True, save_matrices=True, save_imgs=True)