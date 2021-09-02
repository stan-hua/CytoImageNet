"""
Create image crops using bounding box annotations.
"""

import pandas as pd
import numpy as np
import os
from PIL import Image

if "D:\\" not in os.getcwd():
    annotations_dir = "/home/stan/cytoimagenet/annotations/"
    data_dir = '/ferrero/stan_data/'
else:
    annotations_dir = "/annotations/"
    data_dir = 'M:/ferrero/stan_data/'

dir_name = "bbbc041"

df = pd.read_csv(f"{annotations_dir}datasets_info.csv")
useful_cols = ["database", "name", "organism", "cell_type",
               "cell_component", "phenotype", "channels", "microscopy",
               "dir_name"]
row = df[df.dir_name == dir_name][useful_cols]


# Load Metadata containing Bounding Box
df_labels = pd.concat([pd.read_json(f"{data_dir}/{dir_name}/malaria/training.json"),
                       pd.read_json(f"{data_dir}/{dir_name}/malaria/test.json")])
df_labels.reset_index(drop=True, inplace=True)
df_labels["path_file"] = df_labels.image.map(lambda x: x["pathname"])

df_metadata = pd.DataFrame(columns=useful_cols + ["path", "filename"])

# Create processed images dir
try:
    os.mkdir(f"{data_dir}/{dir_name}/crops")
except:
    pass
crop_folder = f"{data_dir}/{dir_name}/crops/"


def get_crops(x):
    global df_metadata
    i_filepath = f"{data_dir}/{dir_name}/malaria" + x["path_file"]
    img = np.array(Image.open(i_filepath))
    # Convert to Grayscale             # Average along channels
    img = img.mean(axis=-1)
    # Crop Number
    img_num = 0

    for annotation in x["objects"]:
        img_num += 1

        # Get Crop
        x_start = annotation["bounding_box"]["minimum"]["r"]
        x_end = annotation["bounding_box"]["maximum"]["r"]
        y_start = annotation["bounding_box"]["minimum"]["c"]
        y_end = annotation["bounding_box"]["maximum"]["c"]
        img_crop = img[x_start:x_end, y_start:y_end]

        # Save Crop
        new_name = x["path_file"].split("/")[2].split(".")[0] + f"_{img_num}.png"
        Image.fromarray(img_crop).convert("L").save(crop_folder + new_name)

        # Add metadata for crop
        label = annotation["category"].lower()
        new_row = row.copy()
        new_row["cell_type"] = label
        new_row["path"] = crop_folder
        new_row["filename"] = new_name
        # df_metadata = pd.concat([df_metadata, new_row], ignore_index=True)


df_labels.apply(get_crops, axis=1)
# df_metadata.to_csv(f"{annotations_dir}/{dir_name}_metadata.csv", index=False)
