from prepare_dataset import check_exists

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import random

import os
import glob


def check_label_files():
    """Return dataframe containing current information on images associated with
    labels.
        - label: cytoimagenet label
        - num_images: number of images assigned to label
        - all_exist: boolean if all images (for a label) exist
    """
    labels = []
    num_examples = []
    all_exists = []
    for file in glob.glob("M:/home/stan/cytoimagenet/annotations/classes/*.csv"):
        labels.append(file.split("classes\\")[-1].replace(".csv", ""))

        df = pd.read_csv(file)
        all_exists.append(all(df.apply(lambda x: True if os.path.exists(x.path + "/" + x.filename) else False, axis=1)))
        num_examples.append(len(df))
    df = pd.DataFrame({"label": labels, "num_images":num_examples, "all_exist":all_exists})
    df.label = df.label.map(lambda x: x.split("classes\\")[-1].replace(".csv", ""))
    return df


def load_images_from(label: str):
    """Return list of arrays (of images) for <label>.
    """
    df = pd.read_csv(f"/home/stan/cytoimagenet/annotations/classes/{label}.csv")
    df.apply(check_exists, axis=1)

    imgs = []
    for i in df.index:
        try:
            im = Image.open(df.loc[i, "path"] + "/" + df.loc[i, "filename"])
            imgs.append(np.array(im.resize((224, 224))))
        except:
            pass
    return imgs


def gridplot_images(imgs: list, label:str):
    # Randomly sample 100
    try:
        img_samples = random.sample(imgs, 25)
    except:
        return False

    # settings
    h, w = 5, 5        # for raster image
    nrows, ncols = 5, 5  # array of sub-plots
    figsize = [6, 8]     # figure size, inches

    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # plot simple raster image on each sub-plot
    n = 0
    for i, axi in enumerate(ax.flat):
        axi.imshow(img_samples[n], cmap="gray")
        axi.set_axis_off()
        n += 1
    fig.suptitle(label)
    plt.savefig(f"/home/stan/cytoimagenet/figures/class_grid_show/{label}_grid.png")


if __name__ == "__main__":
    labels = ["ifn-gamma",
              "factor-x",
              "nfkb pathway inhibitor",
              "s8645",
              "uv inactivated sars-cov-2"]

    labels_2 = ["cell membrane",
                "hpsi0813i-ffdm_3",
                "osteopontin",
                "pik3ca targeted",
                "bacteria",
                "voltage-gated sodium channel blocker",
                "s501357"
                ]

    for label in labels_2:
        # try:
        imgs = load_images_from(label)
        plt.savefig(f"/home/stan/cytoimagenet/figures/class_grid_show/example.png")
        if gridplot_images(imgs, label) is None:
            print("Success! for " + label)
        else:
            print("No images for " + label)
        # except:
        #     print("Failed")
