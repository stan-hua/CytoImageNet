import dask.dataframe as dd
import dask
import pandas as pd
import numpy as np

import json
import os
import glob
import time
from copy import copy

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

sns.set_style("white")

if "D:\\" not in os.getcwd():
    annotations_dir = "/home/stan/cytoimagenet/annotations/"
    data_dir = '/ferrero/stan_data/'
    plot_dir = "/home/stan/cytoimagenet/figures/classes/"
else:
    annotations_dir = "M:/home/stan/cytoimagenet/annotations/"
    data_dir = 'M:/ferrero/stan_data/'
    plot_dir = "D:/projects/cytoimagenet/figures/classes/"


def get_df_metadata() -> dd.DataFrame:
    """Return lazy dask dataframe containing reference to all *_metadata.csv
    in /home/stan/cytoimagenet/annotations/clean/
    """
    return dd.read_csv(f"{annotations_dir}clean/*_metadata.csv",
                       dtype={"organism": "object",
                              "cell_type": "object",
                              "cell_visible": "object",
                              "phenotype": "object",
                              "gene": "object",
                              "sirna": "object",
                              "compound": "object",
                              "microscopy": "category",
                              "idx": "object"
                              })


def get_df_counts() -> pd.DataFrame:
    df_counts = []
    for file in glob.glob(f"{annotations_dir}class_counts/*"):
        df_counts.append(pd.read_csv(file))
    df_counts = pd.concat(df_counts)
    df_counts.rename(columns={"Unnamed: 0": "label"},  inplace=True)
    df_counts.reset_index(drop=True, inplace=True)

    idx = (df_counts.label.value_counts() > 1)
    duplicate = idx[idx].index

    for label in duplicate:
        idx_dup = df_counts[df_counts.label == label].duplicated("counts")
        if len(idx_dup[idx_dup]) > 0:      # duplicate label in two cols
            df_counts.drop(idx_dup[idx_dup].index, inplace=True)
    # [print(df_counts[df_counts.label == i]) for i in idx[idx].index.tolist()]

    return df_counts


def save_counts():
    """
    Group dataframe rows by metadata columns
        - compound
        - sirna
        - gene
        - phenotype
        - cell component
        - cell type
        - organism

    Save unique labels and their row count inclusively. Ignores overlap between
    labels and rows.
    """
    df_ = get_df_metadata()

    cols = ["compound", "sirna", "gene", "phenotype", "cell_visible", "cell_type", "organism"]

    print(f"Shape: ({len(df_)}, {len(df_.columns)})")

    for i in cols:
        df = copy(df_)
        df[i] = df[i].str.split("|")
        df = df.explode(i)
        df_col = df.loc[:, i].value_counts().to_frame().rename(columns={i: "counts"}).compute()

        num_datasets = df.groupby(i)["name"].nunique().compute()
        num_microscopy = df.groupby(i)["microscopy"].nunique().compute()

        df_col["num_datasets"] = df_col.index.map(lambda x: num_datasets[x])
        df_col["num_microscopy"] = df_col.index.map(lambda x: num_microscopy[x])
        df_col["category"] = i
        df_col.to_csv(f"{annotations_dir}/class_counts/{i}_counts.csv")

        print(f"{i} finished!")


def check_existing_classes() -> list:
    """Return list of finished labels."""
    files = glob.glob(annotations_dir + "classes/*.csv")
    return [file.replace(annotations_dir + "classes/", "").replace(".csv", "") for file in files]


def select_classes_uniquely() -> None:
    """ Select unique rows for each label passing the threshold. Saves
    dataframe for each label.

    Algorithm:
        - Iterate through labels whose image count > 400 from saved count dataframe.
        - Filter full metadata dataframe for the label and not 'used'
        - Save rows assigned to the label.
            - Stratified sampling
            - Marks rows used as 'used' in full metadata dataframe
    """
    # Minimum Number of Images Threshold
    thresh = 287

    # Get metadata dataframe
    df_metadata = get_df_metadata().compute()

    # Load in used_indices if available
    if os.path.exists(f"{annotations_dir}classes/used_images.json"):
        with open(f"{annotations_dir}classes/used_images.json") as f:
            used_indices = json.load(f)
    else:
        # Create var to indicate if image (index) is already part of a label
        used_indices = {}

    print("Beginning to collect classes!")
    # Get label counts
    df_counts = get_df_counts()
    df_counts.sort_values(by="counts", inplace=True)
    df_counts = df_counts[df_counts.counts >= thresh].reset_index(drop=True)

    # Check for existing classes
    done_labels = check_existing_classes()

    # Select unique classes
    n = 0
    for i in df_counts.index:
        label = df_counts.loc[i, "label"]
        col = df_counts.loc[i, "category"]

        if label in done_labels:
            print(f"{label} already created!")
            continue

        # Track code runtime
        n += 1
        start = time.perf_counter()

        save_class(df_metadata, col, label, used_indices,
                   thresh=thresh)

        # Analyze code runtime
        simul_time = time.perf_counter() - start
        print(f"Saving {label} took {simul_time} seconds.")
        print(f"Expected Time to Finish: {simul_time * (len(df_counts) - n) / 60} minutes")

        # Update json file
        with open(f"{annotations_dir}classes/used_images.json", 'w') as f:
            json.dump(used_indices, f)


def save_class(df, col: str, label: str, used_indices: dict, thresh=287) -> None:
    """Save rows with <label> in <col> in a dataframe corresponding to label.

    If the label has <= 1000 examples, save filtered dataframe as is, and return
        dataframe where <col> != <label>.

    If the label has >1000 examples, do the following:
        - GroupBy other columns, then sample 1000 rows

    Afterwards,update original <df> with selected rows as 'used' = 1


    :param df: dd.DataFrame containing image metadata
    :param col: category name from dd.DataFrame that contains the unique label
    :param label: value in <col> to be used as a class
    :param used_indices: dictionary of unique image indices already used.
    :param num_datasets: optional number of datasets for this label
    """
    # Filter for unused rows with label
    remove_used = ~df["idx"].isin(used_indices)
    contains_label = df[col].str.contains(label)
    df_filtered = df[(contains_label) & (remove_used)]

    # Get number of examples
    num_examples = len(df_filtered)
    # If # of rows < 287, remove
    if num_examples < thresh:
        print(f"{col} label: {label} has <287 rows!")
        return

    # If # of rows <= 1000, save
    if num_examples <= 1000:
        df_filtered = df_filtered
        df_filtered.to_csv(f"{annotations_dir}/classes/{label}.csv", index=False)
    else:
        cols = ['organism', 'cell_type', 'cell_visible', 'sirna', 'compound', 'phenotype']
        if col != "gene":
            cols.remove(col)

        # If # of rows > 10000, preliminary stratified sampling to 10000 rows
        if num_examples > 10000:
            # downsample by dataset name, organism and cell type (excluding col)
            frac_to_sample = 10000 / num_examples
            to_downsample_by = ["name", "organism", "cell_type"]

            # Remove columns already used to groupby for next groupby operation
            if col in to_downsample_by:
                to_downsample_by.remove(col)
            for used_col in to_downsample_by:
                if used_col != "name":
                    cols.remove(used_col)

            df_filtered = df_filtered.groupby(to_downsample_by, dropna=False, group_keys=False, sort=False).apply(
                lambda x: x.sample(frac=frac_to_sample),)
                # meta={'database': 'str', 'name': 'str', 'organism': 'object',
                #       'cell_type': 'object', 'cell_visible': 'object',
                #       'phenotype': 'object', 'channels': 'str', 'microscopy': 'str',
                #       'dir_name': 'str', 'path': 'str', 'filename': 'str',
                #       'gene': 'object', 'sirna': 'object', 'compound': 'object',
                #       'idx': 'object'})

        if col == "sirna":
            cols.remove("compound")
        elif col == "compound":
            cols.remove("sirna")

        # Stratified sample 1000 rows from 10000+
        latest_num_examples = len(df_filtered)
        print(f"With > 1000 rows, {label} has {latest_num_examples}")

        frac_to_sample_2 = 1000 / latest_num_examples

        draft_filtered = df_filtered.groupby(
            cols, dropna=False, group_keys=False, sort=False).apply(
            lambda x: x.sample(frac=frac_to_sample_2))

        while len(draft_filtered) < 1000:
            if len(draft_filtered) < 500:
                frac_to_sample_2 += 0.03
            else:
                frac_to_sample_2 += 0.02

            draft_filtered = df_filtered.groupby(
                cols, dropna=False, group_keys=False, sort=False).apply(
                lambda x: x.sample(frac=frac_to_sample_2))

        # If still over 1000, randomly sample to get exactly 1000
        if len(draft_filtered) > 1000:
            draft_filtered = draft_filtered.sample(n=1000)

        df_filtered = draft_filtered
        print(f"{label} has {len(df_filtered)} rows after last sampling.")


        df_filtered.to_csv(f"{annotations_dir}/classes/{label}.csv", index=False)

    # Add indices used to dictionary
    used_indices.update(dict.fromkeys(df_filtered["idx"], label))


def remove_classes_with_dir(dir_name: str) -> None:
    """Remove classes with <dir_name>.
        - Remove csv file
        - Update json file to remove indexers
    """
    # Get Clean Metadata
    df_metadata = pd.read_csv(f"{annotations_dir}/clean/{dir_name}_metadata.csv")

    # Get Image Index to Label mapping
    with open(f"{annotations_dir}classes/used_images.json") as f:
        used_indices = json.load(f)

    df_idx_label = pd.Series(used_indices).reset_index().rename(columns={"index": "idx", 0: "label"})
    # Get dir_name
    df_idx_label["dir_name"] = df_idx_label.idx.map(lambda x: x.split("-")[0])

    # Get labels which contain dataset
    labels = df_idx_label[df_idx_label["dir_name"] == dir_name].label.unique()

    # Loop through label csvs with dataset <dir_name>
    for label in labels:
        # Delete label csv file
        os.remove(f"{annotations_dir}/classes/{label}.csv")
        # Remove indexers with label assignment
        remove_indexers = df_idx_label[df_idx_label["label"] == label].idx
        [used_indices.pop(i) for i in remove_indexers]

    with open(f"{annotations_dir}classes/used_images.json", 'w') as f:
        json.dump(used_indices, f)


def remove_class(label: str) -> None:
    """Remove <label>.csv and update used_indices.json
    """

    # Get Image Index to Label mapping
    with open(f"{annotations_dir}classes/used_images.json") as f:
        used_indices = json.load(f)

    df_idx_label = pd.Series(used_indices).reset_index().rename(columns={"index": "idx", 0: "label"})
    # Get dir_name
    df_idx_label["dir_name"] = df_idx_label.idx.map(lambda x: x.split("-")[0])
    [used_indices.pop(i, None) for i in df_idx_label[df_idx_label.label == label].idx.tolist()]

    try:
        os.remove(f"{annotations_dir}/classes/{label}.csv")
    except:
        pass

    with open(f"{annotations_dir}classes/used_images.json", 'w') as f:
        json.dump(used_indices, f)


def plot_class_count():
    df_counts = get_df_counts()
    df_counts.sort_values(by="counts", ascending=False, inplace=True)
    df_counts["plot_counts"] = df_counts["counts"].map(lambda x: x if x <= 2000 else 2000)
    df_counts.category = df_counts.category.map(lambda x: "cell_visible" if x == "cell_visible" else x)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    sns.stripplot(data=df_counts, x="plot_counts", y="category", jitter=True, orient="h", ax=ax)
    ax.set(xlim=(0, 2000), xlabel="Number of Images", ylabel="Labels by Category")
    plt.axvline(x=287, color="gray", linestyle="--")
    plt.savefig(f"{plot_dir}category_labels_vs_num_images.png")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax = sns.scatterplot(data=df_counts, x="plot_counts", y="num_datasets", hue="category")
    ax.set(xlim=(-0.5, 2010), xlabel="Number of Images", ylabel="Number of Datasets")
    plt.legend(loc='upper center')
    plt.savefig(f"{plot_dir}labels_vs_num_dataset.png")
    #
    # fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    # ax = sns.scatterplot(data=df_counts, x="plot_counts", y="num_microscopy", hue="category")
    # ax.set(xlim=(-0.5, 2010), xlabel="Number of Images", ylabel="Number of Microscopy")
    # plt.legend(loc='upper center')


def plot_threshold():
    df_counts = get_df_counts()

    values = []
    for i in range(100, 1000):
        values.append((df_counts.counts >= i).sum())

    df_thresh = pd.DataFrame({"threshold": range(100, 1000),
                              "num_labels": values})
    df_thresh = df_thresh[df_thresh.num_labels <= 1100].reset_index(drop=True)

    ax = sns.scatterplot(data=df_thresh, x="threshold", y="num_labels")
    ax.set(xlabel="# of Images Threshold", ylabel="# of Labels")
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    # now plot both limits against eachother
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    df_diff = abs(df_thresh.threshold - df_thresh.num_labels)
    print(df_thresh[df_diff == df_diff.min()])


if __name__ == "__main__" and "D:\\" not in os.getcwd():
    # save_counts()
    for file in glob.glob(f"{annotations_dir}classes/*.csv"):
        df = pd.read_csv(file)
        if len(df) <= 0:
            os.remove(file)
            print(f"Removed {file}")
        # label = file.split("classes/")[-1].replace(".csv", "")
        # remove_class(label)
        # print(f"Removed {label}")

    select_classes_uniquely()

    print(f"Success!")
else:
    pass
    # plot_class_count()
    # df_counts = get_df_counts()
