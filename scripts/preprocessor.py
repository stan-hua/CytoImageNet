from typing import Optional

import javabridge
import bioformats as bf

import pandas as pd
import numpy as np
import os
from PIL import Image

javabridge.start_vm(class_path=bf.JARS)

if "D:\\" not in os.getcwd():
    annotations_dir = "/home/stan/cytoimagenet/annotations/"
    data_dir = '/ferrero/stan_data/'
else:
    annotations_dir = "D:/projects/cytoimagenet/annotations/"
    data_dir = 'M:/ferrero/stan_data/'


dir_name = "rec_rxrx19b"


def exists_meta(dir_name: str) -> Optional[pd.DataFrame]:
    """Return True if metadata file exists and False otherwise."""
    try:
        return pd.read_csv(f"{annotations_dir}{dir_name}_metadata.csv")
    except:
        raise Exception("Does not exist!")


def load_image(x) -> np.array:
    return bf.load_image(x)


def save_img(x, name, folder_name="merged") -> None:
    if not os.path.isdir(f"{data_dir}{dir_name}/{folder_name}"):
        os.mkdir(f"{data_dir}{dir_name}/{folder_name}")

    Image.fromarray(x).convert("L").save(f"{data_dir}{dir_name}/{folder_name}/{name}")


def merger(paths: list, filenames: list, new_filename: str) -> np.array:
    """Given list of paths + filenames to merge, do the following:
        1. Load in all images at paths
        2. Normalize each photo from 0 to 1 by dividing by maximum intensity of each channel
        3. Add all images and divide by number of images
        4. Save new image as PNG named <new_filename> in "merged" folder
    """
    img_stack = None
    for i in range(len(filenames)):
        img = load_image(paths[i] + "/" + filenames[i])
        img = img/img.max()
        if type(img_stack) == type(None):
            img_stack = img
        else:
            img_stack = np.vstack([img_stack, img])

    # Average along stack & Normalize to 0-255
    img_stack = img_stack.mean(axis=0) * 255

    save_img(img_stack, new_filename)


def slicer(img, x: tuple, y: tuple) -> np.array:
    """Return sliced image.
        -   <x> is a tuple of x_min and x_max.
        -   <y> is a tuple of y_min and y_max.
    """
    return img[x[0]:x[1], y[0]:y[1]]


def save_crops(x):
    img = load_image(f"{x.path}/{x.filename}")
    img_crop = slicer(img, (0, 715), (0, 825))

    save_img(img_crop, x.new_name, "crop")



if __name__ == "__main__":
    # Get and Prepare Given Metadata
    df_labels = pd.read_csv(f"{data_dir}{dir_name}/rxrx19b/metadata.csv")

    def create_path(x):
        return f"{data_dir}{dir_name}/rxrx19b/images/{x.experiment}/Plate{x.plate}"

    df_labels["path"] = df_labels.apply(create_path, axis=1)
    df_labels["filename"] = df_labels.apply(lambda x: f"{x.well}_s{x.site}", axis=1)

    for i in df_labels.index:
        old_paths = [df_labels.loc[i, "path"]] * 6
        old_names = [f"{df_labels.loc[i, 'filename']}_w{i}.png" for i in range(1,7)]
        new_name = f"{df_labels.loc[i, 'experiment']}_{df_labels.loc[i, 'plate']}_{df_labels.loc[i, 'filename']}.png"

        if not os.path.isfile(f"{data_dir}{dir_name}/merged/{new_name}"):
            merger(old_paths, old_names, new_name)
        else:
            print("File exists!")

javabridge.kill_vm()
