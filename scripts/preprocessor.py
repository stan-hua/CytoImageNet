from typing import Optional

import os
import glob
import time

import pandas as pd
import numpy as np
from PIL import Image


if "D:\\" not in os.getcwd():
    annotations_dir = "/home/stan/cytoimagenet/annotations/"
    data_dir = '/ferrero/stan_data/'
else:
    annotations_dir = "D:/projects/cytoimagenet/annotations/"
    data_dir = 'M:/ferrero/stan_data/'


dir_name = "None"


def exists_meta(dir_name: str) -> Optional[pd.DataFrame]:
    """Return True if metadata file exists and False otherwise."""
    try:
        return pd.read_csv(f"{annotations_dir}{dir_name}_metadata.csv")
    except:
        raise Exception("Does not exist!")


def load_image(x) -> np.array:
    return np.array(Image.open(x))


def save_img(x: str, name: str, dir_name: str, folder_name="merged") -> None:
    """Save image at '/ferrero/stan_data/'<dir_name>/<folder_name>/<name>'
        - <dir_name> refers to directory name of dataset
        - <name> refers to new filename
    """
    if not os.path.isdir(f"{data_dir}{dir_name}/{folder_name}"):
        os.mkdir(f"{data_dir}{dir_name}/{folder_name}")

    Image.fromarray(x).convert("L").save(f"{data_dir}{dir_name}/{folder_name}/{name}")


def merger(paths: list, filenames: list, new_filename: str, dir_name: str = dir_name) -> np.array:
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

    save_img(img_stack, new_filename, dir_name=dir_name)


def slicer(img, x: tuple, y: tuple) -> np.array:
    """Return sliced image.
        -   <x> is a tuple of x_min and x_max.
        -   <y> is a tuple of y_min and y_max.
    """
    return img[x[0]:x[1], y[0]:y[1]]


def save_crops(x):
    img = load_image(f"{x.path}/{x.filename}")
    img_crop = slicer(img, (0, 715), (0, 825))

    save_img(img_crop, x.new_name, x.dir_name, "crop")


def create_image(x):
    name = x.filename
    if x.dir_name == "rec_rxrx19a":
        old_paths = [f"{data_dir}{x.dir_name}/RxRx19a/images/{name.split('_')[0]}/Plate{name.split('_')[1]}/"] * 5
        old_names = ["_".join(name.split('_')[2:]).replace(".png", f"_w{chan}.png") for chan in range(1,6)]
    elif x.dir_name == "rec_rxrx19b":
        old_paths = [f"{data_dir}{x.dir_name}/rxrx19b/images/{name.split('_')[0]}/Plate{name.split('_')[1]}/"] * 6
        old_names = ["_".join(name.split('_')[2:]).replace(".png", f"_w{chan}.png") for chan in range(1,7)]
    elif x.dir_name == "idr0093":
        old_paths = [f"{data_dir}{x.dir_name}/{'/'.join(name.split('^')[:-1])}"] * 5
        old_names = [name.split("^")[-1][:-4] + str(i) + ".tif" for i in range(1, 6)]
    elif x.dir_name == "idr0088":
        old_paths = [f"{data_dir}{x.dir_name}/20200722-awss3/ds_hcs_02/PhenoPrintScreen/raw_images_for_IDR/" + "/".join(name.split("^")[:-1])] * 3
        old_names = [name.split("^")[-1].replace("A01", f"A0{i}").replace(".png", f"{i}.tif") for i in range(1, 4)]
    elif x.dir_name == "idr0080":
        old_paths = [f"{data_dir}{x.dir_name}/images/{'/'.join(name.split('^')[:-1])}"] * 5
        old_names = [name.split("^")[-1].replace(".png", f"-ch{i}sk1fk1fl1.tiff") for i in range(1,6)]
    elif x.dir_name == "idr0081":
        old_paths = [f"{data_dir}{x.dir_name}/" + "/".join(name.split("^")[:-1])] * 2
        old_names = [name.split("^")[-1][:-4] + str(i) + ".tif" for i in [1, 2]]
    elif x.dir_name == "idr0003":
        old_paths = [f"{data_dir}{x.dir_name}/201301120/Images/" + "/".join(name.split("_")[:2])] * 2
        old_names = [name.split("_")[2].replace(".png", chan) for chan in ["--GFP.tif", "--Cherry.tif"]]
    elif x.dir_name == "idr0009":
        old_paths = [f"{data_dir}{x.dir_name}/20150507-VSVG/VSVG/" + "/".join(name.split("^")[:-1])] * 3
        old_names = [name.split("^")[-1].replace(".png", f"--{ch}.tif") for ch in ["nucleus-dapi", "pm-647", "vsvg-cfp"]]
    elif x.dir_name == "idr0016":
        old_paths = [f"{data_dir}{x.dir_name}/{'/'.join(name.split('^')[:-1])}-{ch}" for ch in ["Mito", "Hoechst", "ERSytoBleed", "ERSyto", "Ph_golgi"]]
        old_names = []
        for p in old_paths:  # channels have different directories
            _old_name = name.split('^')[-1].replace('.png', '')
            file = glob.glob(f"{p}/{_old_name}*")[0]
            old_names.append(_old_name + file.split(_old_name)[-1])
    elif x.dir_name == "bbbc022":  # TODO: Change filename to contain all 6 names
        df_labels = pd.read_csv(f"{data_dir}{x.dir_name}/BBBC022_v1_image.csv", error_bad_lines=False)
        row = df_labels[df_labels["Image_FileName_OrigHoechst"] == name.replace(".png", ".tif")]
        old_paths = [f"{data_dir}{x.dir_name}/BBBC022_v1_images_{row['Image_Metadata_PlateID']}w{g}" for g in range(1, 6)]
        old_names = row.iloc[:, 1:6].values.flatten().tolist()
    # elif x.dir_name == "bbbc021":  # bbbc021 is excluded for testing
    #     files = glob.glob(f"{data_dir}{x.dir_name}/" + name.replace("^", "/").replace(".png", ""))
    #     old_paths = [p.split("/")[:-1] for p in files]
    #     old_names = [f.split("/")[-1] for f in files]
    else:
        raise NotImplementedError(f"{x.dir_name} merging not implemented!")

    merger(old_paths, old_names, name, x.dir_name)

if __name__ == "__main__":
    # Get and Prepare Given Metadata
    df_labels = pd.read_csv(f"{data_dir}{dir_name}/rxrx2/metadata.csv")

    df_labels["path"] = df_labels.apply(lambda x: f"{data_dir}{dir_name}/rxrx2/images/{x.experiment}/Plate{x.plate}", axis=1)
    df_labels["filename"] = df_labels.apply(lambda x: f"{x.well}_s{x.site}", axis=1)

    n = 1

    for i in df_labels.index:
        start = time.perf_counter()

        old_paths = [df_labels.loc[i, "path"]] * 6
        old_names = [f"{df_labels.loc[i, 'filename']}_w{i}.png" for i in range(1,7)]
        new_name = f"{df_labels.loc[i, 'experiment']}_{df_labels.loc[i, 'plate']}_{df_labels.loc[i, 'filename']}.png"

        if not os.path.isfile(f"{data_dir}{dir_name}/merged/{new_name}"):
            merger(old_paths, old_names, new_name)
        else:
            print("File exists!")

        one_cycle = (time.perf_counter() - start)

        n += 1
        print(f"Progress: {round(100*n / len(df_labels))}%")
        print(f"Time Remaining: {one_cycle * (len(df_labels) - n) / 60: .2f} minutes")









