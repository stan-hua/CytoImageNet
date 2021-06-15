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

dir_name = "bbbc022"


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
    df_metadata = exists_meta(dir_name)
    df_metadata.to_csv(f"{annotations_dir}/{dir_name}_metadata_old.csv", index=False)
    df_metadata.cell_component = "nucleus|er|nucleoli|actin|golgi|mitochondria"
    df_metadata.channels = "f_nucleus|f_er|f_nucleoli|f_actin|f_golgi|f_mitochondria"

    df_labels = pd.read_csv(f"{data_dir}{'bbbc022'}/BBBC022_v1_image.csv", error_bad_lines=False)

    for i in df_labels.index:
        name = "_".join(df_labels.loc[i, "Image_FileName_OrigER"].split("_")[:3])
        idx = df_metadata.filename.str.contains(name)
        old_paths = df_metadata.loc[idx]["path"].tolist()
        old_names = df_metadata.loc[idx]["filename"].tolist()

        merger(old_paths, old_names, name + ".png")

    # Fix filenames
    df_metadata.filename = df_metadata.filename.map(lambda x: "_".join(x.split("_")[:3]) + ".png")
    df_metadata = df_metadata.drop_duplicates("filename")

    # Fix file path
    df_metadata.path = f"{data_dir}{dir_name}/merged"
    df_metadata.to_csv(f"{annotations_dir}/{dir_name}_metadata.csv", index=False)

javabridge.kill_vm()
