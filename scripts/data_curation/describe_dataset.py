"""
Formatting Tips:
    - Separate possible options for some category by "|" (e.g. cytoplasm|actin)
        - Both can be found in the same image
    - For separable channel possibilities, use "/" (e.g. f_actin/f_mitosis
        refers to the channel being either actin or mitosis stained channel)
    - Channels should be labeled by microscopy type, except if fluorescent. If
so, "f_" followed by component stained for (e.g. f_nucleus)
    - Type in lowercase when possible
    - Avoid using hyphens (e.g. wild-type => wildtype)
    - Use singular version of nouns
    - For categories with the same number of items as class sizes given,
        - assume greater importance going down (Phenotype has precedence)

"""

from typing import List, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import os


# HELPER FUNCTION: Return None if object evaluates to False
def empty_input(x): return x if bool(x) else None


# HELPER FUNCTION: Return list if str has elements separated by ","
def contains_list(x) -> Union[List, str]:
    """If <x> represents a list, convert to list of str/int. If not, return <x>.
    """
    if (isinstance(x, float)) or ("," not in x) or ("[" in x):
        return x

    try:  # if int
        assert isinstance(int(x.split(",")[0]), int)
        return [int(i) for i in x.split(",")]
    except:  # if str
        return x.split(", ")


# HELPER FUNCTION
def str_to_eval(x):
    """Evaluates string to convert to integer or list of type str/int, if
    applicable.

    Examples:
    >>> str_to_eval("hello")
    "hello"
    >>> str_to_eval("['hello', 'my', 'name']")
    ['hello', 'my', 'name']
    >>> str_to_eval("[1, 2, 3]")
    [1, 2, 3]
    """
    try:
        return ast.literal_eval(x)
    except:
        return x


# Get input on dataset information
def get_annotations(df: pd.DataFrame) -> None:
    """Get user input to create annotations of datasets."""
    row = pd.Series()
    try:
        row["database"] = empty_input(input("Database: "))
        row["name"] = empty_input(input("Name of Dataset: "))
        # Check if already exists
        if row["name"] in df.name.tolist():
            raise BaseException("Record already exists!")
        row["dir_name"] = empty_input(input("Abbreviated name (e.g. "
                                            "<database>_<dataset>): "))
        row["link"] = empty_input(input("External Link: "))
        row["download"] = empty_input(contains_list(input("Download Link: ")))
        row["metadata_download"] = empty_input(
            contains_list(input("Metadata Download Link: ")))

        row["organism"] = input("Organism: ")
        row["cell_type"] = empty_input(contains_list(input("Cell Type: ")))
        row["cell_component"] = empty_input(contains_list(input("Component: ")))
        row["phenotype"] = empty_input(contains_list(input("Phenotype: ")))

        row["segmented?"] = empty_input(bool(input("Segmented? (Y): ")))
        row["augmented?"] = empty_input(bool(input("Augmented? (Y): ")))

        row["class_sizes"] = empty_input(contains_list(input("Class Sizes: ")))
        row["dataset_size"] = empty_input(input("Number of Fields/Images: "))    # NOTE: 4 channels belong to 1 field
        row["height"] = empty_input(input("Avg. Height: "))
        row["width"] = empty_input(input("Avg. Width: "))
        row["format"] = empty_input(input("Image Format: "))

        row["channels"] = empty_input(contains_list(input("Channels: ")))         # What the channels stain for (e.g. "F_<label>" for fluorescence, "Bright field", "dark field"
        row["microscopy"] = empty_input(contains_list(input("Microscopy: ")))
        row["num_plates"] = empty_input(input("Num Plates: "))

        row["notes"] = input("NOTES: ")

        for col in ["dataset_size", "height", "width", "num_plates"]:
            if row[col] is not None and isinstance(row[col], str):
                row[col] = int(row[col])

        save = bool(input("Save? (Y)"))
        if save:
            df = df.append(row, ignore_index=True)
            df.to_csv("datasets_info.csv", index=False)
    except:
        pass

    if bool(input("Continue? (Y)")):
        get_annotations(df)


# Analyze current dataset annotations
def analyze_datasets(df_: pd.DataFrame, by: str) -> None:
    """Provide estimate for the number of images per class <by> some category.

    ==Representation Invariant==
        <by> must be one of the following: organism, cell_type, cell_component,
    phenotype, or channels.
    """
    df = df_.copy().applymap(str_to_eval)  # evaluate strings
    col = by.lower().strip()  # preprocess input <by>

    # Error-Handling
    if col not in ["organism", "cell_type", "cell_component", "phenotype",
                   "channels"]:
        raise Exception("'by' must be one of the following: organism, "
                        "cell_type, cell_component, phenotype, channels")
    # Null values
    df.dropna(subset=[col], inplace=True)

    # Number of channels
    df["num_channels"] = df.channels.map(
        lambda x: len(x) if isinstance(x, list) else 1)

    if col == "organism":
        # Remove uncertain
        # df[col] = df.loc[:, col].str.replace(" \(\?\)", "")
        # df[col] = df.loc[:, col].str.replace("mus musculus", "mouse")
        # df[col] = df.loc[:, col].str.replace("drosophila", "fly")
        counts = df.dataset_size * df.num_channels
        df_counts = pd.DataFrame({col: df.organism, "num_images": counts})
        df_counts = df_counts.groupby("organism").sum()
    elif col == "cell_type":
        num_cols = df["cell_type"].map(
            lambda x: len(x) if isinstance(x, list) else 1)
        num_classes = df.class_sizes.map(
            lambda x: len(x) if isinstance(x, list) else 1)

        df_counts = pd.DataFrame(columns=["cell_type", "num_images"])

        # For rows whose # of classes == # of cell types
        idx = (num_cols == num_classes)
        df_cell_label = df.loc[idx].reset_index(drop=True)
        for i in range(len(df_cell_label)):
            if isinstance(df_cell_label.loc[i, "cell_type"], list):
                cell_listed = df_cell_label.loc[i, "cell_type"]
                num_listed = np.array(df_cell_label.loc[i, "class_sizes"]) * \
                    df_cell_label.loc[i, "num_channels"]
            else:
                cell_listed = [df_cell_label.loc[i, "cell_type"]]
                num_listed = [df_cell_label.loc[i, "class_sizes"] *
                              df_cell_label.loc[i, "num_channels"]]
            new_row = pd.DataFrame(
                {"cell_type": cell_listed,
                 "num_images": num_listed})
            df_counts = pd.concat([df_counts, new_row])

        df_unlabeled = df.loc[~idx].reset_index(drop=True)
        df_unlabeled = df_unlabeled.loc[df_unlabeled.cell_type.map(len) == 1]

        for i in range(len(df_unlabeled)):
            new_row = pd.DataFrame(
                {"cell_type": [df_unlabeled.loc[i, "cell_type"]],
                 "num_images": [[df_unlabeled.loc[i, "dataset_size"] *
                               df_unlabeled.loc[i, "num_channels"]]]})
            df_counts = pd.concat([df_counts, new_row])

        df_counts = df_counts.groupby(col).sum()


    plot_counts(df_counts)


def plot_counts(df_counts: pd.DataFrame) -> None:
    """Create plots of index (class) and number of images.

    <df_counts>: pd.DataFrame
        df_counts.index contains class names
        df_counts.values contains number of images in the class
    """
    my_cmap = plt.get_cmap("Set3")
    fig, ax = plt.subplots(1, 1)
    bar_plot = ax.bar(range(len(df_counts)), df_counts.values.flatten(),
                      log=True,
                      color=my_cmap(1*np.array(range(len(df_counts)))))
    ax.set_xticks(range(len(df_counts)))
    ax.set_xticklabels(df_counts.index.tolist(), rotation="vertical")
    plt.ylabel("Counts")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Change working directory
    home_dir = "//"
    os.chdir(f"{home_dir}annotations")

    # Load Existing Info
    try:
        df = pd.read_csv("datasets_info.csv")
    except FileNotFoundError:
        df = pd.DataFrame(
            columns=['database', 'name', 'dir_name', 'link', 'download',
                     'metadata_download', 'organism',
                     'cell_type', 'cell_component', 'phenotype', 'segmented?',
                     'class_sizes', 'dataset_size', 'width', 'height', 'format',
                     'channels', 'microscopy', 'num_plates', 'notes'])
    # get_annotations(df)

