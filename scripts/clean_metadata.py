from describe_dataset import str_to_eval, contains_list, empty_input
from typing import Union, Tuple, List
import os
import pandas as pd
import webbrowser

if True:
    annotations_dir = "/home/stan/cytoimagenet/annotations/"
    data_dir = '/ferrero/stan_data/'
else:
    annotations_dir = "D:/projects/cytoimagenet/annotations/"
    data_dir = 'M:/ferrero/stan_data/'

df = pd.read_csv(f"{annotations_dir}datasets_info.csv")
annotation_sets = os.listdir(annotations_dir)
dl_sets = os.listdir(data_dir)


def manual_fix_filenames():
    """Get user input to fix metadata filenames if names do not match available
    data directory names."""
    global df, annotations_dir, annotation_sets
    for name in annotation_sets:
        if sum(df.dir_name.map(lambda x: x in name)) == 0:
            if name == "datasets_info.csv":
                continue

            print(f"Invalid Name! {name}")
            new_name = input("New filename: ")
            os.system(f"mv {annotations_dir}{name} {annotations_dir}{new_name}")
            print("\n\n")


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


def create_metadata(dir_name: str) -> None:
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

    def label_cycle(x):
        if "interphase" in x:
            return "interphase"
        else:
            return "mitosis"

    df_metadata.phenotype = df_metadata.path.map(label_cycle)

    df_metadata.to_csv(f"{annotations_dir}/{dir_name}_metadata.csv",
                       index=False)
    return

    print(row.iloc[0])
    # # TODO: Infer labels from folder name
    # print("Subdirectories: " + str(os.listdir(data_dir + dir_name)))
    # if bool(input("Infer label from subdirectory? (Y) ")):
    #     for label in os.listdir(data_dir + dir_name):
    #         idx = df_metadata.path.str.contains(label)
    #         idx = idx[idx]      # only True
    #         df_metadata.loc[idx.index, "class"] = label
    #     if bool(input("Change inferred labels? (Y)")):
    #         dict_mapping = {}
    #         for label in os.listdir(data_dir + dir_name):
    #             change_label = input(f"Map {label} to ")
    #             dict_mapping[label] = change_label
    #         df_metadata["class"] = df_metadata["class"].map(dict_mapping)
    # else:
    #     col_to_use = input("Use column: ")
    #     while col_to_use not in useful_cols:  # Error Handling
    #         col_to_use = input("Use column: ")
    #
    #     # if brightfield, use organism + cell_type as label
    #     if col_to_use == "microscopy" and \
    #             row.iloc[0]["microscopy"] == "brightfield":
    #         label = row.iloc[0]["organism"]
    #         if isinstance(row.iloc[0]["cell_type"], str):
    #             label += f"-{row.iloc[0]['cell_type']}"
    #
    #     # TODO: Consider cases where other columns are used
    #
    #     df_metadata["class"] = label


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
            pic_format = ['.flex', '.bmp', '.tif', '.png', '.jpg', '.jpeg']
            if any([i in file for i in pic_format]):  # filters for picture formats
                file_paths.append(root.replace("\\","/"))
                file_names.append(file)

    return file_paths, file_names


def exists_meta(dir_name: str) -> None:
    """Return True if metadata file exists and False otherwise."""
    global annotation_sets
    return any([f"{dir_name}_metadata" in meta for meta in annotation_sets])


def rename_bbbc():
    global dl_sets, df
    for name in dl_sets:
        if "bbbc" in name:
            row = df[df.dir_name == name]
            webbrowser.open(row.link.iloc[0])

            new_name = "bbbc" + input("BBBC<name>: ").lower()

            df.loc[row.index[0], 'dir_name'] = new_name
            os.system(f"mv {data_dir}{name} {data_dir}{new_name}")
            df.to_csv(f"{annotations_dir}datasets_info.csv", index=False)



if __name__ == "__main__":
    dir_name = "kag_cell_cycle"
    # create_metadata(dir_name)
else:
    df_metadata = pd.read_csv(f"M:/home/stan/cytoimagenet/annotations/{dir_name}_metadata.csv")
