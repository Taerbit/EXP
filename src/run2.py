import Controller
import tensorflow as tf
import time
import efficientnet.tfkeras

# Run pre-loaded pipelines
start_time= time.time()

#E0 - G
model = tf.keras.models.load_model("..\\models\\200324_EfficientNetB0NoisyStudent_001.h5", compile=False)

image_tags = ["C:\\Users\\finnt\\OneDrive\\Documents\\Uni\\Year 4\\Honours\\Project\\Lesions\\", ".jpg", "ISIC_", "_downsampled", 0]
seg_tags = ["C:\\Users\\finnt\\OneDrive\\Documents\\Uni\\Year 4\\Honours\\Project\\Segmentations\\", ".png", "ISIC_", "_segmentation", 0]
numpy_tags = ["C:\\Users\\finnt\\Documents\\Honours Results\\200324_EfficientNetB0NoisyStudent_001\\grad_cam\\", "_sorted.npy", "ISIC_", "_downsampled", 0]
label_tag = ["C:\\Users\\finnt\\OneDrive\\Documents\\Uni\\Year 4\\Honours\\Project\\Dual Classifier\\", ".csv", "ISIC_", "_downsampled", 0]

input_tags = [image_tags, numpy_tags, seg_tags, label_tag]

output_dir = "C:\\Users\\finnt\\Documents\\Honours Results\\200324_EfficientNetB0NoisyStudent_001\\grad_cam\\"

Controller.pre_loaded_shap(model, input_tags, output=output_dir, save_csv=True)

#E0 - S
model = tf.keras.models.load_model("..\\models\\200324_EfficientNetB0NoisyStudent_001.h5", compile=False)

image_tags = ["C:\\Users\\finnt\\OneDrive\\Documents\\Uni\\Year 4\\Honours\\Project\\Lesions\\", ".jpg", "ISIC_", "_downsampled", 0]
seg_tags = ["C:\\Users\\finnt\\OneDrive\\Documents\\Uni\\Year 4\\Honours\\Project\\Segmentations\\", ".png", "ISIC_", "_segmentation", 0]
numpy_tags = ["C:\\Users\\finnt\\Documents\\Honours Results\\200324_EfficientNetB0NoisyStudent_001\\shap\\", "_sorted.npy", "ISIC_", "_downsampled", 0]
label_tag = ["C:\\Users\\finnt\\OneDrive\\Documents\\Uni\\Year 4\\Honours\\Project\\Dual Classifier\\", ".csv", "ISIC_", "_downsampled", 0]

input_tags = [image_tags, numpy_tags, seg_tags, label_tag]

output_dir = "C:\\Users\\finnt\\Documents\\Honours Results\\200324_EfficientNetB0NoisyStudent_001\\shap\\"

Controller.pre_loaded_shap(model, input_tags, output=output_dir, save_csv=True)

#E7 - G
model = tf.keras.models.load_model("..\\models\\200411_EfficientNetB7NoisyStudent_001.h5", compile=False)

image_tags = ["C:\\Users\\finnt\\OneDrive\\Documents\\Uni\\Year 4\\Honours\\Project\\Lesions\\", ".jpg", "ISIC_", "_downsampled", 0]
seg_tags = ["C:\\Users\\finnt\\OneDrive\\Documents\\Uni\\Year 4\\Honours\\Project\\Segmentations\\", ".png", "ISIC_", "_segmentation", 0]
numpy_tags = ["C:\\Users\\finnt\\Documents\\Honours Results\\200411_EfficientNetB7NoisyStudent_001\\grad_cam\\", "_sorted.npy", "ISIC_", "_downsampled", 0]
label_tag = ["C:\\Users\\finnt\\OneDrive\\Documents\\Uni\\Year 4\\Honours\\Project\\Dual Classifier\\", ".csv", "ISIC_", "_downsampled", 0]

input_tags = [image_tags, numpy_tags, seg_tags, label_tag]

output_dir = "C:\\Users\\finnt\\Documents\\Honours Results\\200411_EfficientNetB7NoisyStudent_001\\grad_cam\\"

Controller.pre_loaded_shap(model, input_tags, output=output_dir, save_csv=True)

print("Finished: " + str((time.time()-start_time)/60) + " mins")