from preprocessor import create_image

import glob
import os
import time

import pandas as pd
import numpy as np
from PIL import Image


def check_croppable(x):
    """Return True if image at absolute file path <x> satisfies the property:
        - Has width or height at least 2 times (224)
    """
    # Ideal image shape for model training
    ideal_shape = (224, 224)

    # Load Image as array
    img = np.array(Image.open(x.path + x.filename))
    return (img.shape[0] >= 2 * ideal_shape[0]) or (
            img.shape[1] >= 2 * ideal_shape[1])


def check_exists(x):
    """Return True if image exists at <x>.

    If False, use dataset-specific method to create image. Raise Exception if
    image creation failed.
    """
    if os.path.exists(x.path + "/" + x.filename):
        return True

    # If file does not exist
    print(f"Creating images for {x.dir_name}")
    start = time.perf_counter()
    create_image(x)
    print(f"Image Created in {time.perf_counter()-start} seconds!")
    return False


def create_crops(x):
    pass


if __name__ == '__main__':
    annotations_dir = "/home/stan/cytoimagenet/annotations/"
    plot_dir = "/home/stan/cytoimagenet/figures/classes/"

    for file in glob.glob(annotations_dir + "classes/*.csv"):
        df = pd.read_csv(file)

        label = file.replace(annotations_dir + "classes/", "").replace(".csv", "")
        # Check if images exist. If not, create images.
        exists_series = df.apply(check_exists, axis=1)

        if all(exists_series):
            print(f"Success! All images present for {label}")
        else:
            print(f"Failed! Not all images are present for {label}!")
        # Skip cropping for labels with >= 1000 images
        if len(df) >= 1000:
            continue

        # Check if images can be cropped to increase number of images
        # df["croppable"] = df.apply(check_croppable)

        # Create & Save Crops. Update dataframe with new crops as new 'examples'
        # df = create_crops(df)

        # Save updated dataframe
        # df.drop(columns=["croppable", "exists"], inplace=True)
        # df.to_parquet(file, index=False)






