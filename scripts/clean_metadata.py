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

dir_name = "rec_rxrx2"


def create_metadata(dir_name: str = dir_name) -> None:
    global df, data_dir, annotations_dir

    # Columns for Metadata
    # useful_cols = ["database", "name", "organism", "cell_type",
    #                "cell_component", "phenotype", "channels", "microscopy",
    #                "dir_name"]
    # row = df[df.dir_name == dir_name][useful_cols]
    # # Get file paths and names | Merge with row
    # data_paths, data_names = get_data_paths(dir_name)
    # row["path"] = None
    # row["path"] = row["path"].astype("object")
    #
    # row.cell_component = 'nucleus|actin'
    # row.channels = "f_nucleus|f_actin|f_mitosis"
    # row.drop(columns=["phenotype"], inplace=True)
    #
    # row.at[row.index[0], "path"] = data_paths
    # df_metadata = row.explode("path", ignore_index=True)
    # df_metadata["filename"] = data_names

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
            if "merged" not in root:
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
        if "bbbc021" in filename:  # Keep BBBC021 for external validation
            filenames.remove(filename)

    if not os.path.exists(f"{annotations_dir}clean/"):
        os.mkdir(f"{annotations_dir}clean/")
    d6tstack.combine_csv.CombinerCSV(filenames,
                                     add_filename=False).to_csv_align(
        output_dir=f"{annotations_dir}clean/",
        output_prefix="", write_params={'index': False})


def fix_labels():
    files = glob.glob(f"{annotations_dir}/*_metadata.csv")

    for file in files:
        gc.collect()
        df = pd.read_csv(file, dtype={"cell_component": "category"})

        # df["microscopy"] = df["microscopy"].str.replace(" \(\?\)", "")
        # df["cell_component"] = df["cell_component"].map(lambda x: "nucleus" if x == "nuclei" else x)
        # df["phenotype"] = df["phenotype"].map(lambda x: None if x == "dmso" else x)
        try:
            col = "compound"
            label = "\*"
            proteins = ["golgin84", "cytb5", "bik", "dtmd-vamp1", "mao", "galt",
                        "pts-1", "rab5", "rab11", "kinase", "calr-kdel",
                        "vamp5", "cco", "ergic53", "rab5a"]
            to = " -- "
            if df[col].str.lower().str.contains(label).sum() > 0:
                df[col] = df[col].str.lower().str.replace(label, to)
                # df[col] = df[col].map(lambda x: to if x == label else x)
                df.to_csv(file, index=False)
                print(f"Success! {col}: {label} -> {to}")
                print(df.iloc[0].dir_name + f" contains {label} in {col}")

            col_2 = "phenotype"
            label_2 = "\*"
            to_2 = " -- "
            if df[col_2].str.lower().str.contains(label_2).sum() > 0:
                df[col_2] = df[col_2].str.lower().str.replace(label_2, to_2)
                # df[col] = df[col].map(lambda x: to if x == label else x)
                df.to_csv(file, index=False)
                print(f"Success! {col}: {label_2} -> {to_2}")
                print(df.iloc[0].dir_name + f" contains {label_2} in {col_2}")
        except:
            pass


def fix_gene():
    """Add ' targeted' to gene labels of all metadata dataframes to distinguish
    gene target from protein visible.
    """
    files = glob.glob(f"{annotations_dir}/clean/*_metadata.csv")
    for file in files:
        gc.collect()
        df = pd.read_csv(file, dtype={"cell_component": "category"})
        df.drop(columns=["Unnamed: 0"], inplace=True)

        df["gene"] = df["gene"].map(lambda x: x + " targeted" if isinstance(x,
                                                                            str) and 'targeted' not in x else x)
        df.to_csv(file, index=False)
    print("Successful! " + f"' targeted' added to gene!")


def add_indexer():
    """Add indexer for all image metadata, following convention:
        - '<dir_name>_<index>'
    """
    files = glob.glob(f"{annotations_dir}/clean/*_metadata.csv")
    for file in files:
        gc.collect()
        df = pd.read_csv(file)
        df["idx"] = df.loc[0, 'dir_name'] + "-" + df.index.astype('str')
        df.to_csv(file, index=False)
        print(f"Success! Indexer added to {df.loc[0, 'dir_name']}")


if __name__ == "__main__" and "D:\\" not in os.getcwd():
    # create_metadata(dir_name)
    # print(f"{dir_name} metadata creation successful!")
    fix_labels()
    fix_column_headers()
    fix_gene()
    add_indexer()
else:
    df_metadata = exists_meta(dir_name)
    if type(df_metadata) is not type(None):
        print(df_metadata.iloc[0])
