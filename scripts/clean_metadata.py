from describe_dataset import str_to_eval, contains_list, empty_input
from typing import Union, Tuple, List, Optional
import os
import pandas as pd
import numpy as np
import webbrowser
import random

if "D:\\" not in os.getcwd():
    annotations_dir = "/home/stan/cytoimagenet/annotations/"
    data_dir = '/ferrero/stan_data/'
else:
    annotations_dir = "D:/projects/cytoimagenet/annotations/"
    data_dir = 'M:/ferrero/stan_data/'

df = pd.read_csv(f"{annotations_dir}datasets_info.csv")
annotation_sets = os.listdir(annotations_dir)
dl_sets = os.listdir(data_dir)

dir_name = "rec_rxrx1"


def create_metadata(dir_name: str = dir_name) -> None:
    global df, data_dir, annotations_dir

    # Columns for Metadata
    useful_cols = ["database", "name", "organism", "cell_type",
                   "cell_component", "phenotype", "channels", "microscopy",
                   "dir_name"]
    row = df[df.dir_name == dir_name][useful_cols]

    # Get file paths and names | Merge with row
    data_paths, data_names = get_data_paths(dir_name)
    row["path"] = None
    row["path"] = row["path"].astype("object")
    row.at[row.index[0], "path"] = data_paths
    df_metadata = row.explode("path", ignore_index=True)
    df_metadata["filename"] = data_names
    col_mapper = {"Characteristics [Organism]": "organism", "Comment [Cell Line]": "cell_type",
                  "Gene Symbol": "gene", "Channels": "channels"
                  }
    df_labels = pd.read_csv(f"{data_dir}{dir_name}/idr0080-screenA-annotation.csv")
    df_labels.rename(columns=col_mapper, inplace=True)

    df_metadata.cell_type = "dma1"   # assign most frequent since it's not possible to find indexer
    # df_metadata.cell_component = df_metadata.path.map(label_component)
    df_metadata.channels = df_metadata.cell_component.map(lambda x: "f_" + x)

    df_metadata.to_csv(f"{annotations_dir}/{dir_name}_metadata.csv",
                       index=False)


def metadownload(metadata: Union[str, list], dir_name: str) -> None:
    """Download metadata with the following convention:
        - <dir_name>_metadata.csv
        - if more than one, <dir_name>_metadata_(num occurence).csv
    """
    global annotations_dir
    # Update known files in <annotations_dir>
    annotation_sets = os.listdir(annotations_dir)
    ann_series = pd.Series(annotation_sets)

    if isinstance(metadata, list):
        for i in metadata:
            metadownload(i, dir_name)

    # Base Case: if metadata is str
    # Create filename
    filename = f"{annotations_dir}{dir_name}_metadata"
    while sum(ann_series.str.contains("filename")) > 0:
        # Get number in filename
        proc_number = filename.split('_')[-1].strip()
        if proc_number in "0123456789":
            filename = f"{annotations_dir}{dir_name}_metadata_" \
                       f"{str(int(proc_number) + 1)}"
        else:  # if no number in filename yet
            filename = f"{annotations_dir}{dir_name}_metadata_1"
    # Add file format
    filename += f".{metadata.split('.')[-1]}"

    # Download Metadata
    os.system(f"wget {metadata} {annotations_dir}")
    # Change filename to match convention
    os.system(f"mv {annotations_dir}{metadata} {annotations_dir}{filename}")


def get_metadata():
    """Download metadata (if available). Otherwise, ask for user input
        - If no metadata available in /annotations
        - dataset downloaded in /ferrero
    """
    global annotation_sets, df, data_dir

    ann_series = pd.Series(annotation_sets)  # convert filename list to Series

    for i in df.index:
        metadata_name = df.loc[i, "dir_name"] + "metadata"
        # If metadata not downloaded AND dataset downloaded
        if sum(ann_series.str.contains(metadata_name)) < 1 and \
                df.loc[i, "dir_name"] in dl_sets:

            # if there is a metadata dl link
            if not isinstance(df.loc[i, "metadata_download"], float) and \
                df.loc[i, "metadata_download"] is not None:
                metadownload(df.loc[i, 'metadata_download'],
                             df.loc[i, 'dir_name'])

            # ask user input on where to download metadata
            elif os.listdir(data_dir + df.loc[i, "dir_name"]):
                webbrowser.open(df.loc[i, "link"])
                new_meta_link = contains_list(
                    empty_input(input("Metadata Download Link/s")))
                df.loc[i, "metadata_download"] = new_meta_link

                if new_meta_link is not None:
                    metadownload(new_meta_link, df.loc[i, "dir_name"])
                else:
                    if bool(input("Create Metadata?")):
                        create_metadata(df.loc[i, "dir_name"])

    # Save changes to df
    df.to_csv(f"{annotations_dir}datasets_info.csv", index=False)


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
            pic_format = ['.flex', '.bmp', '.tif', '.png', '.jpg', '.jpeg', ".BMP", ".Bmp", ".JPG", ".TIF", ".DIB", ".dib", ".dv"]
            if any([i in file for i in pic_format]):  # filters for picture formats
                file_paths.append(root.replace("\\","/"))
                file_names.append(file)

    return file_paths, file_names


def exists_meta(dir_name: str) -> Optional[pd.DataFrame]:
    """Return True if metadata file exists and False otherwise."""
    try:
        return pd.read_csv(f"M:/home/stan/cytoimagenet/annotations/{dir_name}_metadata.csv")
    except:
        print("Does not exist!")



if __name__ == "__main__" and "D:\\" not in os.getcwd():
    create_metadata(dir_name)
    print(f"{dir_name} metadata creation successful!")
else:
    df_metadata = exists_meta(dir_name)
    if type(df_metadata) is not type(None):
        print(df_metadata.iloc[0])


def utos(x):
    if isinstance(x, str):
        if "u-2 os" in x.lower() or "u2-os" in x.lower():
            return "u2os"
        else:
            return x
    elif isinstance(x, list):
        lst_copy = x.copy()
        for i in range(len(lst_copy)):
            lst_copy[i] = utos(lst_copy[i])
        return lst_copy


def try_lower(x):
    if isinstance(x, str):
        return x.lower()
    elif isinstance(x, list):
        lst_copy = x.copy()
        for i in range(len(lst_copy)):
            lst_copy[i] = try_lower(lst_copy[i])

        return lst_copy

