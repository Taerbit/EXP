from datetime import datetime

from EXP.src import Controller, Input, Algorithm, Metrics
import pandas as pd
import numpy as np
import time
import tensorflow as tf
import efficientnet.tfkeras

# Output
o = "" # folder to save shap values and csv to

#Input
grad_cam_csv = ""  # Path to EfficientNetB7 grad cam csv
efficientnetB7 = ""  # Path to EfficientNetB7 model
    # The .jpg, .png and .csv in these locations must only be the ones that are to be used
    # e.g. no other csv files can be in the same destination as the one holding the label.csv
    # ^ It's a hinderance of my loading setup unfortunatley
lesion_image_path = "" # Path to lesion images
segmentation_path = "" # Path to segmentation images
label_csv_path = "" # Path to csv containing labels as index of 0 or 1

# Take a random sampling of 50 correlty predicted images and 50 incorrect predictions of ids from a csv
def get_random_ids():
    df = pd.read_csv(grad_cam_csv)
    randomT = df.loc[df['Correct'] == True].sample(n=50)
    randomF = df.loc[df['Correct'] == False].sample(n=50)
    old_ids = randomT['ID'].tolist() + randomF['ID'].tolist()
    ids = []
    for i in old_ids:
        i = i.replace('ISIC_', '')
        i = i.replace('_downsampled', '')
        i = i.lstrip('0')
        ids.append(i)
    ids.sort(key=int)
    print(ids)
    return ids

# Take a pre used set of ids from a folder
def get_preused_ids():
    paths = Input.get_all_paths("C:\\Users\\finnt\\Documents\\Honours Results\\shap\\", "*.npy")
    del paths[::2] # deletes duplicates
    for i in range(len(paths)):
        paths[i] = paths[i].replace("C:\\Users\\finnt\\Documents\\Honours Results\\shap\\ISIC_", "")
        paths[i] = paths[i].replace("_downsampled", "")
        paths[i] = paths[i].replace("_sorted.npy", "")
        paths[i] = paths[i].lstrip('0')
    return paths



# Retreive ids
ids = get_random_ids()

# set up background
all_imgs = Input.get_all_paths(lesion_image_path, "*.jpg")
# Take 100 random sample images for the background
bg = [all_imgs[i] for i in np.random.randint(0, len(all_imgs), size=5)]
background = []
st = time.time()
print("Start")
for i in range(len(bg)):
    b = Input.load_input_image(bg[i], 300, 225)
    background.append(np.array([b]))
print("Loading Background Dataset Done: " + str((time.time()-st)/60) + " mins")

#load model
model = tf.keras.models.load_model(efficientnetB7, compile=False)
#model.layers[-1].activation=tf.keras.activations.linear

# tags
image_tags = [lesion_image_path, ".jpg", "ISIC_", "_downsampled", 0]
seg_tags = [segmentation_path, ".png", "ISIC_", "_segmentation", 0]
label_tag = [label_csv_path, ".csv", "ISIC_", "_downsampled", 0]

input_tags= [image_tags, seg_tags, label_tag, ids]
time = datetime.now()
Controller.hons(model, input_tags, ["conv5_block32_2_conv"], output=o, save_csv=True, save_matrices=True, background=background)
print(datetime.now()-time)