import Controller
import tensorflow as tf

# Run pre-loaded pipelines

model = tf.keras.models.load_model("../test/test_data/Dual Classifier.h5", compile=False)

image_tags = ["C:\\Users\\finnt\\OneDrive\\Documents\\Uni\\Year 4\\Honours\\Project\\Lesions\\", ".jpg", "ISIC_", "_downsampled", 0]
seg_tags = ["C:\\Users\\finnt\\OneDrive\\Documents\\Uni\\Year 4\\Honours\\Project\\Segmentations\\", ".png", "ISIC_", "_segmentation", 0]
numpy_tags = ["C:\\Users\\finnt\\OneDrive\\Documents\\Uni\\Year 4\\Honours\\Project\\pipeline\\output\\shap\\Dual Classifier\\sorted\\", ".npy", "ISIC_", "_downsampled", 0]
label_tag = ["C:\\Users\\finnt\\OneDrive\\Documents\\Uni\\Year 4\\Honours\\Project\\Dual Classifier\\", ".csv", "ISIC_", "_downsampled", 0]

input_tags = [image_tags, numpy_tags, seg_tags, label_tag]

output_dir = "C:\\Users\\finnt\\OneDrive\\Documents\\Uni\\Year 4\\Honours\\Project\\pipeline\\output\\metric\\shap\\4\\"

Controller.pre_loaded_shap(model, input_tags, output=output_dir, save_csv=True)