from describe_dataset import str_to_eval, contains_list, empty_input
from typing import Union, Tuple, List, Optional
import os
import pandas as pd
import numpy as np
import webbrowser
import random
import glob
import d6tstack.combine_csv
import gc

if "D:\\" not in os.getcwd():
    annotations_dir = "/home/stan/cytoimagenet/annotations/"
    data_dir = '/ferrero/stan_data/'
else:
    annotations_dir = "D:/projects/cytoimagenet/annotations/"
    data_dir = 'M:/ferrero/stan_data/'

df = pd.read_csv(f"{annotations_dir}datasets_info.csv")

dir_name = "kag_bacteria"


def create_metadata(dir_name: str = dir_name) -> None:
    global df, data_dir, annotations_dir

    # Columns for Metadata
    useful_cols = ["database", "name", "organism", "cell_type",
                   "cell_component", "phenotype", "channels", "microscopy",
                   "dir_name"]
    row = df[df.dir_name == dir_name][useful_cols]
    row["phenotype"] = None
    # Get file paths and names | Merge with row
    data_paths, data_names = get_data_paths(dir_name)
    row["path"] = None
    row["path"] = row["path"].astype("object")

    row.cell_component = 'nucleus'
    row.channels = "f_nucleus|brightfield"
    row.microscopy = "fluorescence|brightfield"

    row.at[row.index[0], "path"] = data_paths
    df_metadata = row.explode("path", ignore_index=True)
    df_metadata["filename"] = data_names

    label_map = {0: 'nucleoplasm', 1: "nuclear membrane", 2: "nucleoli",
                 3: "nucleoli", 4: "nuclear speckles", 5: "nuclear bodies",
                 6: "er", 7: "golgi",
                 8: "intermediate filaments", 9: "actin", 10: "microtubules",
                 11: "mitotic spindle", 12: "centrosome", 13: "cell membrane", 14: "mitochondria",
                 15: "aggresome", 16: "cell body", 17: "vesicles", 18: None}

    df_labels = pd.read_csv(f"{data_dir}{dir_name}/train.csv")

    def get_multi_label(x):
        int_labels = [int(g) for g in x.split("|")]
        multi_labels = [label_map[int_label] for int_label in int_labels]

        # If None
        if multi_labels == [None]:
            return None

        # Continue removing if popular labels in multi-label
        num = 0
        while len(multi_labels) > 1 and \
                any([x in multi_labels for x in ["nucleus", "er", "actin",
                                                 "golgi", "nucleoli",
                                                 "mitochondria"]]):
            if "nucleus" in multi_labels:
                multi_labels.remove("nucleus")
                continue
            elif "er" in multi_labels:
                multi_labels.remove("er")
                continue
            elif "golgi" in multi_labels:
                multi_labels.remove("golgi")
                continue
            elif "nucleoli" in multi_labels:
                multi_labels.remove("nucleoli")
                continue
            elif "mitochondria" in multi_labels:
                multi_labels.remove("mitochondria")
                continue
            elif "actin" in multi_labels:
                multi_labels.remove("actin")
                continue

        return "|".join(multi_labels)

    df_labels.Label = df_labels.Label.map(get_multi_label)
    df_labels.set_index("ID", inplace=True)

    def hpa_label(x):
        if "_blue" in x:
            return "nucleus"
        elif "_red" in x:
            return "microtubules"
        elif "_yellow" in x:
            return "er"
        elif "_green" in x:
            try:
                return df_labels.loc[x.split("_green")[0]].Label
            except KeyError:
                return ""

    df_metadata["cell_component"] = df_metadata.filename.map(hpa_label)
    df_metadata["channels"] = df_metadata.cell_component.map(lambda x: f"f_{x}" if isinstance(x, str) and len(x) > 0 else "f_protein")

    df_metadata.to_csv(f"{annotations_dir}/{dir_name}_metadata.csv",
                       index=False)


def print_meta():
    """Download metadata (if available). Otherwise, ask for user input
        - If no metadata available in /annotations
        - dataset downloaded in /ferrero
    """
    global df, dir_name
    to_download = str_to_eval(
        df[df.dir_name == dir_name].metadata_download.iloc[0])
    if isinstance(to_download, str):
        print(f"wget {to_download}")
    else:
        for i in to_download:
            print(f"wget {i}")


def get_data_paths(dir_name: str) -> Tuple[List[str], List[str]]:
    """Return Tuple containing parallel lists containing
        - Path to file
        - Filename
    """
    global data_dir
    file_paths = []
    file_names = []

    for root, dir, files in os.walk(data_dir + dir_name):
        for file in files:
            pic_format = ['.bmp', '.tif', '.png', '.jpg', '.jpeg', ".BMP",
                          ".Bmp", ".JPG", ".TIF", ".DIB", ".dib",
                          ".dv"]  # '.flex' removed
            if any([i in file for i in
                    pic_format]):  # filters for picture formats
                file_paths.append(root.replace("\\", "/"))
                file_names.append(file)

    return file_paths, file_names


def exists_meta(dir_name: str) -> Optional[pd.DataFrame]:
    """Return metadata dataframe if file exists."""
    try:
        return pd.read_csv(
            f"M:/home/stan/cytoimagenet/annotations/{dir_name}_metadata.csv")
    except:
        print("Does not exist!")


def fix_column_headers():
    """ Align *_metadata.csv files to make them easily readable by Dask. Save to
    /home/stan/cytoimagenet/annotations/clean/
    """
    filenames = list(glob.glob(f"{annotations_dir}*_metadata.csv"))
    for filename in filenames:
        if "bbbc021" in filename:
            filenames.remove(filename)

    if not os.path.exists(f"{annotations_dir}clean/"):
        os.mkdir(f"{annotations_dir}clean/")
    d6tstack.combine_csv.CombinerCSV(filenames).to_csv_align(
        output_dir=f"{annotations_dir}clean/")


def fix_labels():
    files = glob.glob(f"{annotations_dir}/*_metadata.csv")

    for file in files:
        gc.collect()
        df = pd.read_csv(file, dtype={"cell_component": "category"})
        # df["microscopy"] = df["microscopy"].str.replace(" \(\?\)", "")
        # df["cell_component"] = df["cell_component"].map(lambda x: "nucleus" if x == "nuclei" else x)
        # df["phenotype"] = df["phenotype"].map(lambda x: None if x == "dmso" else x)
        try:
            col = "cell_type"
            label = "white blood cell \("
            to = "white blood cell"
            if df[col].str.lower().str.contains(label).sum() > 0:
                # df[col] = df[col].str.lower().str.replace(label, to)
                df[col] = df[col].str.lower().map(lambda x: "white blood cell" if "white blood cell \(" in x else x)
                df.to_csv(file, index=False)
                print(f"Success! {col}: {label} -> {to}")
                print(df.iloc[0].dir_name + f" contains {label} in {col}")
        except:
            pass


if __name__ == "__main__" and "D:\\" not in os.getcwd():
    # create_metadata(dir_name)
    # print(f"{dir_name} metadata creation successful!")
    fix_labels()
    fix_column_headers()
else:
    df_metadata = exists_meta(dir_name)
    if type(df_metadata) is not type(None):
        print(df_metadata.iloc[0])
