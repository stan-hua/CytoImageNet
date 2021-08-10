from preprocessor import get_file_references

import os
import glob
import json

import pandas as pd

# PATHS
annotations_dir = "/home/stan/cytoimagenet/annotations/"
plot_dir = "/home/stan/cytoimagenet/figures/classes/"
data_dir = "/ferrero/stan_data/"


def get_image_origin(x):
    """Return list of absolute references to original images (if applicable)
    for each image index (metadata row).

        - Use original image/s retrieved from database (with no preprocessing)
        - Images created from merging multiple images will return a list of all
            channel images
    """
    # If existing image was merged from multiple images
    if x.dir_name in ["rec_rxrx1", "rec_rxrx2", "rec_rxrx19a", "rec_rxrx19b",
                      "idr0093", "idr0088", "idr0080", "idr0081", "idr0003",
                      "idr0009", "idr0016", "bbbc022", "idr0017", "idr0037",
                      "bbbc017"]:
        # Get hard-coded paths and filenames
        old_paths, old_names = get_file_references(x)
        # Verify existence of each file
        to_remove = []
        for k in range(len(old_paths[:])):  # channels have different directories
            if not os.path.exists(f"{old_paths[k]}/{old_names[k]}"):
                to_remove.append(k)
        # Create absolute file paths
        filenames = [old_paths[k] + "/" + old_names[k] for k in range(len(old_paths)) if k not in to_remove]
    elif x.dir_name not in ['bbbc041', 'bbbc045', 'bbbc020', 'idr0067', 'idr0021',
                        'idr0072',]:
        filenames = [x.path + "/" + x.filename]
    elif x.dir_name == "bbbc041":
        filenames = [f"{data_dir}{x.dir_name}/malaria/images/{x.filename.split('_')[0]}.png"]
    elif x.dir_name == "bbbc045":
        if "2014" in x.filename or "2015" in x.filename:
            old_file = f"{data_dir}{x.dir_name}/Stained_Montages" + "/"
            old_file += "_".join(x.filename.split("Stained_Montages_")[1].split("_")[:2]) + "/"
            old_file += "/".join(x.filename.split("Stained_Montages_")[1].split("_")[2:4]) + "/"
            old_file += x.filename.split("_")[-2] + "_"
            old_file += x.filename.split("_")[-1].replace(".png", ".tif")
        else:
            old_file = f"{data_dir}{x.dir_name}/Stained_Montages" + "/" + "/".join(x.filename.split("Stained_Montages_")[1].split("_")[:4]) + "_" + x.filename.split("_")[-1].replace(".png", ".tif")
        filenames = [old_file]
    elif x.dir_name == "bbbc020":
        filenames = [f"{data_dir}{x.dir_name}/{x.filename.split('_')[0]}/{x.filename.replace('.png', '.TIF')}"]
    elif x.dir_name == "idr0067":
        filenames = [x.path + "/" + x.filename.replace(".png", ".dv")]
    elif x.dir_name == "idr0021":
        filenames = [x.path + "/" + x.filename.replace(".png", ".dv")]
    elif x.dir_name == "idr0072":
        filenames = [x.path + "/" + x.filename.replace(".png", ".flex")]
    return filenames


if __name__ == "__main__":
    try:
        with open(f"{annotations_dir}/idx_to_original_images.json") as f:
            all_mapping = json.load(f)
    except:
        all_mapping = {}

    all_used_files = glob.glob(annotations_dir + "classes/*.csv")
    all_used_files.extend(glob.glob(annotations_dir + "unused_classes/*.csv"))
    for file in all_used_files:
        df = pd.read_csv(file)
        df = df[~df.idx.isin(all_mapping)]
        if len(df) > 0:
            filename_series = df.apply(get_image_origin, axis=1)
            curr_mapping = pd.Series(filename_series.values, index=df.idx).to_dict()
            all_mapping.update(curr_mapping)

            # Save Mapping
            with open(f"{annotations_dir}/idx_to_original_images.json", 'w') as f:
                json.dump(all_mapping, f)

        print(file.split("classes/")[-1].replace(".csv", "") + " Done!")
