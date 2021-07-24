from typing import Optional

import os
import glob
import time
import re
import json

import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


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
        return pd.read_csv(f"{annotations_dir}/clean/{dir_name}_metadata.csv")
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
    img_stack = []
    for i in range(len(filenames)):
        if os.path.exists(paths[i] + "/" + filenames[i]):
            img = load_image(paths[i] + "/" + filenames[i])

            # If RGB, convert to grayscale
            if len(img.shape) == 3 and img.shape[-1] == 3:
                img = img.mean(axis=-1)

            # Get 0.1 and 99.9th percentile of pixel intensities
            top_001 = np.percentile(img.flatten(), 99.9)
            bot_001 = np.percentile(img.flatten(), 0.1)

            # Limit maximum intensity to 0.99th percentile
            img[img > top_001] = top_001
            # Floor intensities below 0.1th percentile to 0.
            img[img <= bot_001] = 0
            # Then subtract by the 0.1th percentile intensity
            img = img - bot_001

            # Normalize between 0 and 1
            img = img / img.max()
            img_stack.append(img)
        else:
            print(paths[i] + "/" + filenames[i] + " missing!")

        if len(img_stack) == 0:
            print("Error! No images loaded for: " + paths[i] + "/" + filenames[i])

    img_stack = np.stack(img_stack, axis=-1)

    # Average along stack & Normalize to 0-255
    img_stack = img_stack.mean(axis=-1) * 255

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


def get_file_references(x):
    """Return tuple containing
        - paths to images
        - old file names

    Will be used in merging channels of the 'same' image. Same refers to the
    same plate, well and field but captured for some fluorescent stain.

    NOTE: Dataset-specific methods extract the old path from the current
    filename.
    """
    name = x.filename
    if x.dir_name == "rec_rxrx1":
        old_paths = [f"{data_dir}{x.dir_name}/rxrx1/images/{name.split('_')[0]}/Plate{name.split('_')[1]}/"] * 6
        old_names = ["_".join(name.split('_')[2:]).replace(".png", f"_w{chan}.png") for chan in range(1,7)]
    elif x.dir_name == "rec_rxrx2":
        old_paths = [f"{data_dir}{x.dir_name}/rxrx2/images/{name.split('_')[0]}/Plate{name.split('_')[1]}/"] * 6
        old_names = ["_".join(name.split('_')[2:]).replace(".png", f"_w{chan}.png") for chan in range(1,7)]
    elif x.dir_name == "rec_rxrx19a":
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
        old_names = [re.sub(r"A0[0-9]", f"A0{i}", name.split("^")[-1]).replace(".png", f"{i}.tif") for i in range(1, 4)]
    elif x.dir_name == "idr0080":
        old_paths = [f"{data_dir}{x.dir_name}/{'/'.join(name.split('^')[:-1])}"] * 5
        old_names = [name.split("^")[-1].replace(".png", f"-ch{i}sk1fk1fl1.tiff") for i in range(1,6)]
    elif x.dir_name == "idr0081":
        old_paths = [f"{data_dir}{x.dir_name}/" + "/".join(name.split("^")[:-1])] * 2
        old_names = [name.split("^")[-1][:-4] + f"w{i}" + ".tif" for i in [1, 2]]
    elif x.dir_name == "idr0003":
        if "cell_body" not in x.path:
            old_paths = [f"{data_dir}{x.dir_name}/201301120/Images/" + "/".join(name.split("_")[:2])] * 2
            old_names = [name.split("_")[2].replace(".png", chan) for chan in ["--GFP.tif", "--Cherry.tif"]]
        else:
            df_meta = exists_meta("idr0003")
            df_meta = df_meta[df_meta.filename.str.contains(name.split("--Transmit")[0])]
            another_name = df_meta[df_meta.channels != "brightfield"].iloc[0].filename
            old_paths = [f"{data_dir}{x.dir_name}/201301120/Images/" + "/".join(another_name.split("_")[:2])]
            old_names = [x.filename]
    elif x.dir_name == "idr0009":
        old_paths = [f"{data_dir}{x.dir_name}/20150507-VSVG/VSVG/" + "/".join(name.split("^")[:-1])] * 3
        old_names = [name.split("^")[-1].replace(".png", f"--{ch}.tif") for ch in ["nucleus-dapi", "pm-647", "vsvg-cfp"]]
    elif x.dir_name == "idr0016":
        all_paths = [f"{data_dir}{x.dir_name}/{'/'.join(name.split('^')[:-1])}-{ch}" for ch in ["Mito", "Hoechst", "ERSytoBleed", "ERSyto", "Ph_golgi"]]
        old_names = []
        old_paths = []
        _old_name = name.split('^')[-1].replace('.png', '')
        for k in range(len(all_paths)):  # channels have different directories
            file = glob.glob(f"{all_paths[k]}/{_old_name}*")
            if len(file) >= 1:
                old_names.append(_old_name + file[0].split(_old_name)[-1])
                old_paths.append(all_paths[k])
    elif x.dir_name == "bbbc022":
        try:
            df_labels = pd.read_csv(f"{data_dir}{x.dir_name}/BBBC022_v1_image.csv", error_bad_lines=False)
        except:
            df_labels = pd.read_csv(f"{data_dir}{x.dir_name}/BBBC022_v1_image.csv", on_bad_lines='skip')
        row = df_labels[df_labels["Image_FileName_OrigHoechst"] == name.replace(".png", ".tif")]
        old_paths = [f"{data_dir}{x.dir_name}/BBBC022_v1_images_{row['Image_Metadata_PlateID'].iloc[0]}w{g}" for g in [2, 1, 5, 4, 3]]
        old_names = row.iloc[:, 1:6].values.flatten().tolist()
    elif x.dir_name == "idr0017":
        old_paths = [f"{data_dir}{x.dir_name}/20151124/14_X-Man_10x/source/" + "/".join(name.split("^")[:-1])] * 2
        old_names = [name.split("^")[-1].replace(").png", f" wv {i} - {i}).tif") for i in ["DAPI", "Cy3"]]
    elif x.dir_name == "idr0037":
        old_paths = [f"{data_dir}{x.dir_name}/images/" + "/".join(name.split("^")[:-1])] * 3
        old_names = [name.split("^")[-1].replace(".png", f"-ch{i}sk1fk1fl1.tiff") for i in range(1,4)]
    elif x.dir_name == "bbbc017":
        with open(f"{annotations_dir}/bbbc017_name-path_mapping.json") as f:
            map_name = json.load(f)
        old_paths = [f"{data_dir}{x.dir_name}/" + "/".join(map_name[name].split("^")[:-1])] * 3
        old_names = [name.replace(".png", f"{chan}.DIB") for chan in ["d0", "d1", "d2"]]
    # elif x.dir_name == "bbbc021":  # bbbc021 is excluded for testing
    #     files = glob.glob(f"{data_dir}{x.dir_name}/" + name.replace("^", "/").replace(".png", ""))
    #     old_paths = [p.split("/")[:-1] for p in files]
    #     old_names = [f.split("/")[-1] for f in files]
    else:
        old_paths = None
        old_names = None
        print(f"{x.dir_name} merging not implemented!")

    return old_paths, old_names


def create_image(x):
    """If image does not exist for image associated with metadata row <x>,
    create image by merging channels.

    NOTE: Finding reference to images is dataset-specific.
    """
    name = x.filename
    old_paths, old_names = get_file_references(x)

    # If image not applicable to be merged, early exit
    if old_paths is None:
        return

    if len(old_paths) != len(old_names):
        print(f"Length of Paths != Names for {x.dir_name}")

    # Verify existence of each file
    to_remove = []
    for k in range(len(old_paths[:])):  # channels have different directories
        if not os.path.exists(f"{old_paths[k]}/{old_names[k]}"):
            to_remove.append(k)

    old_paths = [old_paths[k] for k in range(len(old_paths)) if k not in to_remove]
    old_names = [old_names[k] for k in range(len(old_names)) if k not in to_remove]

    if len(old_paths) == 0:
        print(f"Error! No images listed for {x.dir_name} at {old_paths[0]}/{old_names[0]}")
        return

    merger(old_paths, old_names, name, x.dir_name)


if __name__ == "__main__":
    pass









