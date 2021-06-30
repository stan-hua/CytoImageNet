from stratified_sampler import stratified_sample

import dask.dataframe as dd
import pandas as pd
import numpy as np

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
    annotations_dir = "D:/projects/cytoimagenet/annotations/"
    data_dir = 'M:/ferrero/stan_data/'


def get_df_metadata() -> dd.DataFrame:
    """Return lazy dask dataframe containing reference to all *_metadata.csv
    in /home/stan/cytoimagenet/annotations/clean/
    """
    return dd.read_csv(f"{annotations_dir}clean/*_metadata.csv",
                       dtype={"organism": "category",
                              "cell_type": "category",
                              "cell_component": "category",
                              "phenotype": "category",
                              "gene": "category",
                              "sirna": "category",
                              "compound": "category",
                              "microscopy": "category",
                              })


def get_df_counts() -> pd.DataFrame:
    df_counts = []
    for file in glob.glob(f"/home/stan/cytoimagenet/annotations/class_counts/*"):
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

    cols = ["compound", "sirna", "gene", "phenotype", "cell_component", "cell_type", "organism"]

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
    df_metadata = get_df_metadata()

    print("Creating indexer...")

    # Create index for every image
    df_metadata = df_metadata.assign(idx=1)
    df_metadata["idx"] = df_metadata.idx.cumsum() - 1

    # Create variable to indicate if image is already part of a label
    df_metadata["used"] = False
    print("Done!")

    print("Beginning to collect classes!")
    # Get label counts
    df_counts = get_df_counts()
    df_counts.sort_values(by="counts", inplace=True)
    df_counts = df_counts[df_counts.counts >= thresh].reset_index(drop=True)

    # TODO: Select unique classes
    n = 0
    for i in df_counts.index:
        # Track code runtime
        n += 1
        start = time.perf_counter()

        label = df_counts.loc[i, "label"]
        col = df_counts.loc[i, "category"]
        save_class(df_metadata, col, label)

        # Analyze code runtime
        simul_time = time.perf_counter()-start
        print(f"Saving one class took {simul_time} seconds.")
        print(f"Expected Time to Finish: {simul_time * (len(df_counts) - n) / 60} minutes")

        # TODO: Exit Early
        n += 1
        if n == 5:
            print("Early Exit")
            return


def save_class(df, col: str, label: str) -> None:
    """Save rows with <label> in <col> in a dataframe corresponding to label.

    If the label has <= 1000 examples, save filtered dataframe as is, and return
        dataframe where <col> != <label>.

    If the label has >1000 examples, do the following:
        - GroupBy other columns, then sample 1000 rows

    Afterwards,update original <df> with selected rows as 'used' = 1


    :param df: dd.DataFrame containing image metadata
    :param col: category name from dd.DataFrame that contains the unique label
    :param label: value in <col> to be used as a class
    """
    # TODO: Filter for rows with label
    df_filtered = df[(df[col] == label) & (df['used'] == False)]
    num_examples = len(df_filtered)

    # TODO: If # of rows <= 2000, save
    if num_examples <= 1000:
        df_filtered.compute().to_parquet(f"{annotations_dir}/classes/{label}.parquet", index=False)
    else:
        # TODO: Else, group remaining rows by dataset name, organism, cell_type, ...
        cols = ['organism', 'cell_type', 'cell_component', 'phenotype', 'gene', 'sirna', 'compound']
        cols.remove(col)

        # TODO: Stratified sample 1000 rows
        frac_to_sample = 1000 / num_examples
        df_filtered = df_filtered.groupby(cols, dropna=False).sample(frac=frac_to_sample).compute()
        df_filtered.to_parquet(f"{annotations_dir}/classes/{label}.parquet", index=False)

    used_indices = df_filtered["idx"]
    df.map_partitions(lambda df_: df_.assign(used=df_["idx"].isin(used_indices) | df_["used"]))


def plot_class_count():
    df_counts = get_df_counts()
    df_counts.sort_values(by="counts", ascending=False, inplace=True)
    df_counts["plot_counts"] = df_counts["counts"].map(lambda x: x if x <= 2000 else 2000)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    sns.stripplot(data=df_counts, x="plot_counts", y="category", jitter=True, orient="h", ax=ax)
    ax.set(xlim=(0, 2000), xlabel="Number of Images", ylabel="Labels by Category")
    plt.axvline(x=200, color="gray", linestyle="--")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax = sns.scatterplot(data=df_counts, x="plot_counts", y="num_datasets", hue="category")
    ax.set(xlim=(-0.5, 2010), xlabel="Number of Images", ylabel="Number of Datasets")
    plt.legend(loc='upper center')

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax = sns.scatterplot(data=df_counts, x="plot_counts", y="num_microscopy", hue="category")
    ax.set(xlim=(-0.5, 2010), xlabel="Number of Images", ylabel="Number of Microscopy")
    plt.legend(loc='upper center')


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
    select_classes_uniquely()
    print(f"Success!")
else:
    # plot_class_count()
    df_counts = get_df_counts()
