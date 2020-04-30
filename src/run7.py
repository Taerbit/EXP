from EXP import Controller
from tensorflow.keras.models import load_model
import efficientnet.tfkeras
import tensorflow as tf

model ="..\\models\\200324_EfficientNetB0NoisyStudent_001.h5"

output = "C:\\Users\\finnt\\Documents\\Honours Results\\200324_EfficientNetB0NoisyStudent_001\\test\\"

image_tags = ["..\\imgs\\Lesions\\all\\", ".jpg", "ISIC_", "_downsampled", 0]
seg_tags = ["C:\\Users\\finnt\\OneDrive\\Documents\\Uni\\Year 4\\Honours\\Project\\Segmentations\\", ".png", "ISIC_", "_segmentation", 0]
label_tag = ["..\\models\\", ".csv", "ISIC_", "_downsampled", 0]

input_tags = [image_tags, seg_tags, label_tag]


m = load_model(model)
# conv5_block32_2_conv
Controller.hons(m, input_tags, ["top_conv"], output=output, save_csv=True, save_matrices=True, background=[1])