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

dir_name = "rec_rxrx1"


def exists_meta(dir_name: str) -> Optional[pd.DataFrame]:
    """Return True if metadata file exists and False otherwise."""
    try:
        return pd.read_csv(f"{annotations_dir}{dir_name}_metadata.csv")
    except:
        raise Exception("Does not exist!")


if not os.path.isdir(f"{data_dir}{dir_name}/merged"):
    os.mkdir(f"{data_dir}{dir_name}/merged")


def merger(paths: list, filenames: list, new_filename: str) -> None:
    """Given list of paths + filenames to merge, do the following:
        1. Load in all images at paths
        2. Normalize each photo from 0 to 1 by dividing by maximum intensity
        3. Add all images and divide by number of images
        4. Save new image as PNG named <new_filename> in "merged" folder
    """
    img_stack = None
    for i in range(len(filenames)):
        if img_stack is None:
            img_stack = bf.load_image(paths[i] + "/" + filenames[i])
            # Normalize Intensity
            img_stack = img_stack/img_stack.max()
        else:
            img = bf.load_image(paths[i] + "/" + filenames[i])
            # Normalize Intensity by dividing by max
            img = img/img.max()
            img_stack = np.vstack([img_stack, img])

    # Average along stack
    img_stack = img_stack.mean(axis=0)

    # Normalize back to 0-255
    img_stack = img_stack * 255

    Image.fromarray(img_stack).convert("L").save(f"{data_dir}{dir_name}/merged/{new_filename}")



if __name__ == "__main__":
    df = pd.read_csv(f"{annotations_dir}datasets_info.csv")
    useful_cols = ["database", "name", "organism", "cell_type",
                   "cell_component", "phenotype", "channels", "microscopy",
                   "dir_name"]
    row = df[df.dir_name == dir_name][useful_cols]
    row.phenotype = None
    row.channels = "f_nucleus|f_er|f_actin|f_nucleoli|f_mitochondria|f_golgi"

    df_metadata = pd.DataFrame(columns=useful_cols+["sirna", "path", "filename"])

    df_labels = pd.read_csv(f"{data_dir}{dir_name}/rxrx1/metadata.csv")

    def create_path(x):
        return f"{data_dir}{dir_name}/rxrx1/images/{x.experiment}/Plate{x.plate}"

    def create_filename(x):
        return f"{x.well}_s{x.site}"

    df_labels["path"] = df_labels.apply(create_path, axis=1)
    df_labels["filename"] = df_labels.apply(create_filename, axis=1)
    for i in df_labels.index:
        old_paths = [df_labels.loc[i, "path"]] * 6
        old_names = [f"{df_labels.loc[i, 'filename']}_w{i}.png" for i in range(1,7)]
        new_name = f"{df_labels.loc[i, 'experiment']}_{df_labels.loc[i, 'plate']}_{df_labels.loc[i, 'filename']}.png"

        merger(old_paths, old_names, new_name)

        # Add row to metadata
        new_row = row.copy()
        new_row["cell_type"] = df_labels.loc[i, "cell_type"]
        if df_labels.loc[i, "cell_type"] != "EMPTY":
            new_row["sirna"] = df_labels.loc[i, "cell_type"]
        else:   # change EMPTY to no siRNA
            new_row["sirna"] = None
        new_row["filename"] = new_name

        df_metadata = pd.concat([df_metadata, new_row], ignore_index=True)


    # Fix file path
    df_metadata.path = f"{data_dir}{dir_name}/merged"
    df_metadata.to_csv(f"{annotations_dir}/{dir_name}_metadata.csv", index=False)

javabridge.kill_vm()
