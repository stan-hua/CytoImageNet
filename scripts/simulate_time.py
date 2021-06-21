from typing import Optional
import time

import bioformats as bf
import javabridge

import pandas as pd
import numpy as np
import os
from PIL import Image


if "D:\\" not in os.getcwd():
    annotations_dir = "/home/stan/cytoimagenet/annotations/"
    data_dir = '/ferrero/stan_data/'
else:
    annotations_dir = "D:/projects/cytoimagenet/annotations/"
    data_dir = 'M:/ferrero/stan_data/'

dir_name = "idr0072"


def exists_meta(dir_name: str) -> Optional[pd.DataFrame]:
    """Return True if metadata file exists and False otherwise."""
    try:
        return pd.read_csv(f"{annotations_dir}{dir_name}_metadata.csv")
    except:
        raise Exception("Does not exist!")


def load_image(x, loader) -> np.array:
    if loader == "bf":
        return bf.load_image(x)
    else:
        return np.array(Image.open(x))


def merger(paths: list, filenames: list, new_filename: str, loader="bf") -> np.array:
    """Given list of paths + filenames to merge, do the following:
        1. Load in all images at paths
        2. Normalize each photo from 0 to 1 by dividing by maximum intensity of each channel
        3. Add all images and divide by number of images
        4. Save new image as PNG named <new_filename> in "merged" folder
    """
    img_stack = None
    for i in range(len(filenames)):
        img = load_image(paths[i] + "/" + filenames[i], loader)
        img = img/img.max()
        if type(img_stack) == type(None):
            img_stack = img
        else:
            img_stack = np.vstack([img_stack, img])

    # Average along stack & Normalize to 0-255
    img_stack = img_stack.mean(axis=0) * 255


if __name__ == "__main__":
    df_metadata = exists_meta(dir_name)
    df_labels = pd.read_csv(f"{data_dir}{'bbbc022'}/BBBC022_v1_image.csv", error_bad_lines=False)

    i = df_labels.index[0]
    name = "_".join(df_labels.loc[i, "Image_FileName_OrigER"].split("_")[:3])
    folder = df_labels.loc[i, "Image_PathName_OrigER"].replace("/", "")
    idx = (df_metadata.filename.str.contains(name)) & (df_metadata.path.str.contains(folder))
    old_paths = df_metadata.loc[idx]["path"].tolist()
    old_names = df_metadata.loc[idx]["filename"].tolist()


    for loader in ["bf", "PIL"]:

        start = time.perf_counter()
        if loader == "bf":
            javabridge.start_vm(class_path=bf.JARS)
            for i in range(100):
                print(f"BioFormats loading images...")
                merger(old_paths, old_names, name + ".png", loader=loader)
            javabridge.kill_vm()
        else:
            for i in range(100):
                print(f"{loader} loading images...")
                merger(old_paths, old_names, name + ".png", loader=loader)
        end = time.perf_counter()
        print(f"{loader} took {end-start} seconds for 100 operations.")
