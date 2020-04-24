from EXP.src import Controller, Input, Algorithm, Metrics
import pandas as pd
import numpy as np
import time
import tensorflow as tf

# Output
o = "C:\\Users\\finnt\\Documents\\Honours Results\\shap\\linear\\"

# Take a random sort of ids from a csv
def get_random_ids():
    df = pd.read_csv("C:\\Users\\finnt\Documents\\Honours Results\\200416_DenseNet201_001\\grad_cam.csv")
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
ids = get_preused_ids() #get_random_ids()

# set up background
all_imgs = Input.get_all_paths("..\\imgs\\Lesions\\all\\", "*.jpg")
# Take 100 random sample images for the background
bg = [all_imgs[i] for i in np.random.randint(0, len(all_imgs), size=100)]
background = []
st = time.time()
for i in range(len(bg)):
    b = Input.load_input_image(bg[i], 300, 225)
    background.append(np.array([b]))
print("Done: " + str((time.time()-st)/60) + " mins")

#load model
model = tf.keras.models.load_model("..\\models\\200416_DenseNet201_001.h5", compile=False)
model.layers[-1].activation=tf.keras.activations.linear

# tags
image_tags = ["..\\imgs\\Lesions\\all\\", ".jpg", "ISIC_", "_downsampled", 0]
seg_tags = ["C:\\Users\\finnt\\OneDrive\\Documents\\Uni\\Year 4\\Honours\\Project\\Segmentations\\", ".png", "ISIC_", "_segmentation", 0]
label_tag = ["..\\models\\", ".csv", "ISIC_", "_downsampled", 0]

input_tags= [image_tags, seg_tags, label_tag, ids]

Controller.hons(model, input_tags, ["conv5_block32_2_conv"], output=o, save_csv=True, save_matrices=True, background=background)
