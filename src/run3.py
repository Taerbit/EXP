import Controller


image_tags = ["C:\\Users\\finnt\\OneDrive\\Documents\\Uni\\Year 4\\Honours\\Project\\Lesions\\", ".jpg", "ISIC_", "_downsampled", 0]
seg_tags = ["C:\\Users\\finnt\\OneDrive\\Documents\\Uni\\Year 4\\Honours\\Project\\Segmentations\\", ".png", "ISIC_", "_segmentation", 0]
label_tags = ["C:\\Users\\finnt\\OneDrive\\Documents\\Uni\\Year 4\\Honours\\Project\\pipeline\\output\\shap\\Dual Classifier\\", "_sorted.npy",]

input_tags = [image_tags, numpy_tags, seg_tags]

output_dir = "C:\\Users\\finnt\\OneDrive\\Documents\\Uni\\Year 4\\Honours\\Project\\pipeline\\output\\metric\\shap\\1"

Controller.pre_loaded_shap(input_tags, output=output_dir, save_csv=True)