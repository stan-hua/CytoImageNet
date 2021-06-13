from typing import Optional

import javabridge
import bioformats as bf

import pandas as pd
import numpy as np
import os
import png

javabridge.start_vm(class_path=bf.JARS)

data_dir = "/ferrero/stan_data/"
dir_name = "kag_hpa_single"

filenames = os.listdir(data_dir + dir_name + "/train")


def exists_meta(dir_name: str) -> Optional[pd.DataFrame]:
    """Return True if metadata file exists and False otherwise."""
    try:
        return pd.read_csv(f"M:/home/stan/cytoimagenet/annotations/"
                           f"{dir_name}_metadata.csv")
    except:
        print("Does not exist!")


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
            img_stack = bf.load_image(paths[i] + filenames[i])
            # Normalize Intensity
            img_stack = img_stack/img_stack.max()
        else:
            img = bf.load_image(paths[i] + filenames[i])
            # Normalize Intensity by dividing by max
            img = img/img.max()
            img_stack = np.vstack([img_stack,img])

    # Average along stack
    img_stack = img_stack.mean(axis=0)

    # Normalize back to 0-255
    img_stack = img_stack * 255

    if not os.isdir("{data_dir}{dir_name}/merged"):
        os.mkdir("{data_dir}{dir_name}/merged")

    png.from_array(img_stack).save(f"{data_dir}{dir_name}/merged/{new_filename}")




javabridge.kill_vm()
