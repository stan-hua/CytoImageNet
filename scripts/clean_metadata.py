from describe_dataset import str_to_eval, contains_list, empty_input
from typing import Union, Tuple, List, Optional
import os
import pandas as pd
import numpy as np
import webbrowser
import random
import glob
import d6tstack.combine_csv

if "D:\\" not in os.getcwd():
    annotations_dir = "/home/stan/cytoimagenet/annotations/"
    data_dir = '/ferrero/stan_data/'
else:
    annotations_dir = "D:/projects/cytoimagenet/annotations/"
    data_dir = 'M:/ferrero/stan_data/'

df = pd.read_csv(f"{annotations_dir}datasets_info.csv")

to_do = ["idr0016"]

dir_name = to_do[0]

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

    row["channels"] = "f_nucleus|f_protein"
    row["phenotype"] = None
    row.drop(["organism", "cell_type"], axis=1, inplace=True)

    row.at[row.index[0], "path"] = data_paths
    df_metadata = row.explode("path", ignore_index=True)
    df_metadata["filename"] = data_names

    col_mapper = {"Characteristics [Cell Line]": "cell_type",
                  "Characteristics [Organism]": "organism",
                  "Compound Name": "compound", "Compound MoA": "phenotype",
                  "Comment [Gene Symbol]": "gene"
                  }
    df_labels = []
    for f in glob.glob(f"{data_dir}{dir_name}/*.csv"):
        if ".gz" in f:
            df_labels.append(pd.read_csv(f, compression="gzip"))
        else:
            df_labels.append(pd.read_csv(f))
    df_labels = pd.concat(df_labels, ignore_index=True)
    df_labels.rename(columns=col_mapper, inplace=True)

    df_labels.cell_type = df_labels.cell_type.str.lower()
    df_labels.gene = df_labels.gene.str.lower()
    df_labels.organism = df_labels.organism.map(lambda x: "human" if x == "Homo sapiens" else "mouse")
    df_labels["Comment [Protein Name Abbreviation]"] = df_labels["Comment [Protein Name Abbreviation]"].str.lower()

    df_labels.Plate = df_labels.Plate.str.replace("Landmark", "landmarks")

    def get_well(x):
        if "vamp_deg_" in x.lower() or "vamp_degbin" in x.lower() or "vamp1_deg_" in x.lower():
            well = x.split("__")[2]
            return well[0] + str(int(well[1:]))
        elif "ws_nmumg_" in x.lower():
            return None
        else:
            letter = chr(64 + int(x[:3]))
            number = int(x[3:6])
            return f"{letter}{number}"

    def get_plate(x):
        if "WS_" in x.path:
            return "/".join(("WS_" + x.path.split("WS_")[1]).split("/")[:2])
        elif "vamp_deg_" in x.filename.lower() or "vamp_degbin" in x.filename.lower() or "vamp1_deg_" in x.filename.lower():
            return "/".join(x.filename.split("__")[:2])
        elif "VAMP_" in x.path:
            return "/".join(("VAMP_" + x.path.split("VAMP_")[1]).split("/")[:2])

    df_metadata["Plate"] = df_metadata.apply(get_plate, axis=1)
    df_metadata["Well"] = df_metadata.filename.map(get_well)
    df_metadata.dropna(subset=["Well"], inplace=True)

    df_metadata = pd.merge(df_metadata, df_labels, how="left", on=["Plate", "Well"])

    df_metadata.filename = df_metadata.filename.str.replace(".tif", ".png")

    def label_cell_component(x):
        if x["Comment [Protein Name Abbreviation]"] is None or isinstance(x["Comment [Protein Name Abbreviation]"], float):
            return "nucleus"
        return "nucleus|" + x["Comment [Protein Name Abbreviation]"]

    def label_channels(x):
        if x["Comment [Protein Name Abbreviation]"] is None or isinstance(x["Comment [Protein Name Abbreviation]"], float):
            return "f_nucleus"
        return "f_nucleus|f_" + x["Comment [Protein Name Abbreviation]"]

    df_metadata["cell_component"] = df_metadata.apply(label_cell_component, axis=1)
    df_metadata.channels = df_metadata.apply(label_channels, axis=1)

    cols = ['database', 'name', 'organism', 'cell_type', 'cell_component',
            'phenotype', 'gene', 'channels', 'microscopy', 'dir_name',
            'path', 'filename']

    df_metadata.drop(['Plate', 'Well', 'Term Source 1 REF',
                      'Term Source 1 Accession', 'Term Source 2 REF',
                      'Term Source 2 Accession', 'Comment [Gene Identifier]',
                      'Comment [Protein Name]', 'Comment [Protein Name Abbreviation]',
                      'Protein Sequence (aa)', 'UniProt ID', 'Addgene ID',
                      'Experimental Condition [Subcellular Localization]', 'Channels'], axis=1, inplace=True)

    df_metadata.reindex(columns=cols).to_csv(
        f"{annotations_dir}/{dir_name}_metadata.csv", index=False)


# if __name__ == "__main__":
#     main_dir = "/ferrero/stan_data/idr0009/20150507-VSVG/VSVG/"
#
#     for i in df_metadata.index:
#         # Skip if merged file exists
#         if not os.path.exists(f"{data_dir}{dir_name}/merged/{df_metadata.loc[i, 'filename']}"):
#             name = df_metadata.loc[i].filename
#             old_paths = [main_dir + "/".join(name.split("^")[:-1])] * 3
#             old_names = [name.split("^")[-1] + f"--{i}.tif" for i in ["nucleus-dapi", "pm-647", "vsvg-cfp"]]
#             merger(old_paths, old_names, name)



def fix_column_headers():
    filenames = list(glob.glob(f"{annotations_dir}*_metadata.csv"))

    if not os.path.exists(f"{annotations_dir}clean/"):
        os.mkdir(f"{annotations_dir}clean/")
    d6tstack.combine_csv.CombinerCSV(filenames).to_csv_align(output_dir=f"{annotations_dir}clean/")


def print_meta():
    """Download metadata (if available). Otherwise, ask for user input
        - If no metadata available in /annotations
        - dataset downloaded in /ferrero
    """
    global df, dir_name
    to_download = str_to_eval(df[df.dir_name == dir_name].metadata_download.iloc[0])
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
            pic_format = ['.bmp', '.tif', '.png', '.jpg', '.jpeg', ".BMP", ".Bmp", ".JPG", ".TIF", ".DIB", ".dib", ".dv"] # '.flex' removed
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

