from preprocessor import normalize
from typing import Union
from math import ceil, sqrt, floor

import numpy as np
import pandas as pd

from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pylab

import random
import os
import glob

from sklearn.cluster import DBSCAN
import umap
import umap.plot

import torch
import torchvision

if "D:\\" not in os.getcwd():
    annotations_dir = "/home/stan/cytoimagenet/annotations/"
    data_dir = '/ferrero/stan_data/'
    model_dir = "/home/stan/cytoimagenet/model/"
    plot_dir = "/home/stan/cytoimagenet/figures/"
else:
    annotations_dir = "M:/home/stan/cytoimagenet/annotations/"
    data_dir = 'M:/ferrero/stan_data/'
    model_dir = "M:/home/stan/cytoimagenet/model/"
    plot_dir = "M:/home/stan/cytoimagenet/figures/"

sns.set_style("dark")
plt.style.use('dark_background')


def check_label_files():
    """Return dataframe containing current information on images associated with
    labels.
        - label: cytoimagenet label
        - num_images: number of images assigned to label
        - all_exist: boolean if all images (for a label) exist
    """
    labels = []
    num_examples = []
    all_exists = []
    for file in glob.glob(f"{annotations_dir}classes/*.csv"):
        labels.append(file.split("classes\\")[-1].replace(".csv", ""))

        df = pd.read_csv(file)
        if len(df) <= 0:
            print(file)

        all_exists.append(all(df.apply(lambda x: True if os.path.exists(x.path + "/" + x.filename) else False, axis=1)))
        num_examples.append(len(df))
    df = pd.DataFrame({"label": labels, "num_images": num_examples, "all_exist":all_exists})
    df.label = df.label.map(lambda x: x.split("classes\\")[-1].replace(".csv", ""))
    return df


def load_images_from_label(label: str):
    """Return list of arrays (of images) for <label>.
    """
    df = pd.read_csv(f"{annotations_dir}classes/{label}.csv")
    imgs = []
    for i in df.sample(n=25).index:
        try:
            im = Image.open(df.loc[i, "path"] + "/" + df.loc[i, "filename"])
            imgs.append(np.array(im.resize((224, 224))))
        except:
            pass
    return imgs


# GRIDPLOT
def gridplot_images(imgs: list, save=False, fig_title=None, save_name='plot'):
    """Plot grid of images. Use only as many to create a perfectly filled in
     square."""
    num_imgs = len(imgs)
    n_sqrt = floor(sqrt(num_imgs))
    nrows, ncols = n_sqrt, n_sqrt  # array of sub-plots
    figsize = [8, 10]     # figure size, inches

    # Create subplots
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,
                           sharex=True, sharey=True)

    # Plot image on each sub-plot
    n = 0
    for i, axi in enumerate(ax.flat):
        axi.imshow(imgs[n], cmap="gray")
        axi.set_axis_off()
        axi.set_aspect('equal')
        n += 1

    if fig_title is not None:
        fig.suptitle(fig_title)

    # Remove space between subplots
    for j in range(min(100, nrows*ncols)):
        plt.tight_layout(pad=0)

    if save:
        plt.savefig(f"{plot_dir}class_grid_show/{save_name}.png")


def torch_gridplot_images(imgs: list, save=False, fig_title=None, save_name='plot'):
    """Plot grid of images. Use only as many to create a perfectly filled in
     square."""
    num_imgs = len(imgs)
    nrows = floor(sqrt(num_imgs))

    used_imgs = imgs.copy()
    random.shuffle(used_imgs)

    if len(imgs) > nrows ** 2:      # sample images if above perfect square
        used_imgs = random.sample(imgs, nrows ** 2)

    # Convert list of images to tensor.
    # Convert from (num_imgs, H, W, channels) to (num_imgs, channels, H, W)
    imgs_as_tensor = torch.from_numpy(np.stack(used_imgs)).permute(0, 3, 1, 2)
    # Create grid of images
    grid_img = torchvision.utils.make_grid(imgs_as_tensor, nrow=nrows,
                                           padding=0)

    # Plot grayscale gridplot
    plt.axis('off')
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_img.permute(1, 2, 0), cmap='gray')
    if fig_title is not None:
        plt.title(fig_title)
    if save:
        plt.savefig(f"{plot_dir}class_grid_show/{save_name}.png", )


def plot_labels(labels):
    """Create and save gridplots for each label in <labels>.
    """
    for label in labels:
        imgs = load_images_from_label(label)
        if gridplot_images(imgs, fig_title=f"{label}", save=True) is None:
            print("Success! for " + label)
        else:
            print("No images for " + label)


def plot_cytoimagenet():
    df_metadata = pd.read_csv("/ferrero/cytoimagenet/metadata.csv")
    df_metadata = df_metadata.groupby(by=['label']).sample(n=1).reset_index(drop=True)

    imgs = []
    for i in df_metadata.index:
        im = Image.open(df_metadata.loc[i, "path"] + "/" + df_metadata.loc[i, "filename"])
        imgs.append(np.array(im.resize((28, 28))))

    gridplot_images(imgs, save=True, save_name='cytoimagenet_plot')


# CHECK FOR OUTLIERS
def cluster_by_density(embeds):
    clusters = DBSCAN().fit(embeds.iloc[:, :2])
    return clusters.labels_


def is_outlier(embeds: pd.DataFrame):
    """Returns boolean array of same length as <arr>, where True if value is an
    outlier in the context of the rows of the dataframe and False otherwise.

    ==Precondition==:
        - first two columns represent 2D UMap embeddings
        - third column represents labels

    NOTE: Uses definition of outlier given by IQR
    """
    df_embeds = embeds.copy()

    # Cluster points
    clusters = DBSCAN().fit(embeds.iloc[:, :2])
    df_embeds['cluster'] = clusters.labels_
    df_embeds['outlier'] = None

    for cluster in np.unique(clusters.labels_):
        if len(df_embeds[df_embeds.cluster == cluster]['labels'].unique()) == 1 or len(df_embeds[df_embeds.cluster == cluster]) < 100:
            df_embeds.loc[(df_embeds.cluster == cluster), 'outlier'] = True
        else:
            df_embeds.loc[(df_embeds.cluster == cluster), 'outlier'] = False

    return df_embeds.outlier.to_numpy()


# UMAP
def create_umap(labels: Union[str, list],
                directory: str = "imagenet-activations/", kind: str = "",
                data_subset='train'):
    """Return tuple of UMAP 2D embeddings and labels for each row.

    ==Parameters==:
        directory: specifies whether to use activations from randomly initiated
            AlexNet embeddings, or ImageNet pretrained EfficientNetB0.
            - either "imagenet-activations/" or "random_model-activations/"
        name: save figure as <name>.png
    """
    # Accumulate activations & label handle
    activations = []
    label_handle = []
    file_paths = []

    if data_subset != 'val':
        for label in labels:
            # Get all metadata for label
            if kind == "upsampled":
                class_meta_filename = f"{annotations_dir}classes/upsampled/{label}.csv"
            elif kind == "base":
                class_meta_filename = f"{annotations_dir}classes/{label}.csv"
            df_class = pd.read_csv(class_meta_filename).reset_index(drop=True)
            # exists_series = df_class.apply(check_exists, axis=1)
            # df_class = df_class[exists_series]
            # Activations
            temp = pd.read_csv(f"{model_dir}{directory}/{kind}/{label}_activations.csv")
            # num_null = temp.isna().any(axis=1).sum()
            # if num_null > 0 and len(temp) == len(df_class):
            #     print(f"{label} contains {num_null} null values!")
            #     # Filter out NAs
            #     df_class = df_class.loc[~temp.isna().sum(axis=1).map(lambda x: x > 0)]
            #     df_class.to_csv(class_meta_filename, index=False)
            #     print(f"{label} updated!")
            # # Accumulate non-null activations
            # temp = temp.dropna().reset_index(drop=True)
            activations.append(temp)

            # Confirm activations and metadata match
            if len(temp) != len(df_class):
                print(kind, label, " has uneven activation - metadata")
                print("Length Activations/Label Metadata: ", len(temp), len(df_class))

            # Accumulate labels & absolute file paths
            label_handle.extend([label] * len(temp))
            file_paths.extend(df_class.apply(lambda x: x.path + "/" + x.filename, axis=1).tolist())

        activations = pd.concat(activations, ignore_index=True)
    else:   # Get activations for validation set
        activations = pd.read_csv(f"{model_dir}{directory}/unseen_classes_embeddings ({weights}, {dset}).csv")
        df = pd.read_csv("/ferrero/stan_data/unseen_classes/metadata.csv")
        label_handle = df.label.to_numpy()

    # Find 2D U-Map Embeddings
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(activations)

    return embedding, np.array(label_handle), file_paths


def plot_umap(embeds: np.array, labels: list, name: str = "", save: bool = False):
    """Plot 2D U-Map of extracted image embeddings for <labels>.

    ==Parameters==:
        name: save figure as <name>.png
    """
    plt.figure()
    try:
        ax = sns.scatterplot(x=embeds[:, 0], y=embeds[:, 1],
                             hue=labels,
                             legend="full",
                             alpha=1,
                             palette="tab20",
                             s=2,
                             linewidth=0)
    except:
        print(name)
        print("Embed Array Shape: ", embeds.shape)
        print("Num Labels: ", len(labels))
        raise Exception

    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.xlabel("")
    plt.ylabel("")
    plt.tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)
    plt.tight_layout()

    # Save Figure
    if save:
        if not os.path.isdir(f"{plot_dir}umap/"):
            os.mkdir(f"{plot_dir}umap/")
        plt.savefig(f"{plot_dir}umap/{name}.png", bbox_inches='tight', dpi=400)


def plot_umap_by(by: str, kind: str = "base", save: bool = False):
    """
    ==Parameters==:
        - by: one of 'dataset', 'resolution', 'cluster'
        - if kind == 'upsampled', then by can also == 'resolution'
    """
    # Error-Handling
    assert by in ["dataset", "resolution"]
    if by == "resolution" and kind != "upsampled":
        raise Exception("Only upsampled classes can be plotted by resolution!")


    df_embed = pd.read_csv(model_dir + f'imagenet-activations/{kind}_embeddings (random 20, imagenet).csv')
    if by == "cluster":
        # Density-based Clustering
        cluster_labels = cluster_by_density(df_embed)

        # Gridplot 25 images from each cluster
        for cluster in np.unique(cluster_labels):
            embed_cluster = df_embed.loc[cluster_labels == cluster]
            if len(embed_cluster) > 25:
                embed_cluster = embed_cluster.sample(n=25)
            imgs = [np.array(Image.open(path)) for path in embed_cluster.full_path.tolist()]
            gridplot_images(imgs, f"{kind} cluster_{cluster}", save=True)

        # Plot U-Map labeled by cluster assignment
        plot_umap(np.array(df_embed.iloc[:, :2]), cluster_labels, name=f"{kind}/{kind}_clustered", save=save)

    if kind == "upsampled":
        label_dir = f"{annotations_dir}classes/upsampled/"
    else:
        label_dir = f"{annotations_dir}classes/"

    # Get labels from metadata to group 'by'
    info = []
    for label in df_embed.labels.unique():
        df = pd.read_csv(f"{label_dir}/{label}.csv")
        if by == "dataset":
            info.extend(df.name.tolist())
        else:
            info.extend(df.scaling.tolist())

        # Check if embeddings and metadata don't match
        if len(df_embed[df_embed.labels == label]) != len(df):
            print(label)

    # Create Plot
    plt.figure()
    ax = sns.scatterplot(x=df_embed.iloc[:, :2].to_numpy()[:, 0], y=df_embed.iloc[:, :2].to_numpy()[:, 1],
                         hue=info,
                         legend='full',
                         alpha=1,
                         palette="gist_rainbow",
                         s=2,
                         linewidth=0)
    plt.tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)

    if by == "dataset":
        legend_handles = ax.get_legend_handles_labels()
        ax.get_legend().remove()
        # Place legend on separate plot
        fig_2 = plt.figure(figsize=(4.15, 7.3))
        pylab.figlegend(*ax.get_legend_handles_labels(), loc='upper left')

        if save:
            plt.savefig(f"{plot_dir}umap/{kind} (by {by}), legend.png",
                        bbox_inches='tight', dpi=400)

    if save:
        if not os.path.isdir(f"{plot_dir}umap/"):
            os.mkdir(f"{plot_dir}umap/")
        fig = ax.get_figure()
        fig.savefig(f"{plot_dir}umap/{kind} (by {by}).png",
                    bbox_inches='tight', dpi=400)


def from_label_to_paths(label: str, kind: str):
    """Return all image paths for <label> under <kind> preprocessing.

    ==Preconditions==:
        - kind is either 'base' (no modification) or 'upsampled'
    """
    if kind == "upsampled":
        df = pd.read_csv(f"{annotations_dir}classes/upsampled/{label}.csv")
    else:
        df = pd.read_csv(f"{annotations_dir}classes/{label}.csv")
    return df.apply(lambda x: x.path + "/" + x.filename, axis=1).tolist()


def main():
    # Parameters
    weights = "cytoimagenet"      # 'imagenet' or None
    dset = 'toy_50'

    # Directory to load activations
    if weights is None:
        activation_loc = "random_model-activations/"
    elif weights == "cytoimagenet":
        activation_loc = "cytoimagenet-activations/"
    else:
        activation_loc = "imagenet-activations/"

    # Labels to Visualize
    random_classes = ['fgf-20', 'hpsi0513i-golb_2', 'distal convoluted tubule',
                      'fcgammariia', 'pentoxifylline', 'oxybuprocaine', 'il-27',
                      'phospholipase inhibitor', 'estropipate', 'tl-1a',
                      'methacholine', 'cdk inhibitor', 'cobicistat', 'il-28a',
                      'dna synthesis inhibitor', 'lacz targeted',
                      'ccnd1 targeted', 's7902', 'clofarabine', 'ficz']

    # VALIDATION SET
    embeds, labels, full_paths = create_umap(random_classes, directory=activation_loc,
                                             kind=None, data_subset='val')

    # Plot U-Map labeled by category labels
    plot_umap(np.array(embeds), labels, name=f"unseen_classes ({weights}, {dset})", save=True)


if __name__ == "__main__" and "D:\\" not in os.getcwd():
    plot_cytoimagenet()
elif "D:\\" in os.getcwd():
    # df_base = pd.read_csv(model_dir + "similarity/base.csv")
    # df_up = pd.read_csv(model_dir + "similarity/upsampled.csv")
    # df_full = pd.merge(df_base, df_up, on="label")
    # df_full['change_intra_cos'] = df_full["intra_cos_y"] - df_full["intra_cos_x"]
    # df_full['change_inter_cos'] = df_full["inter_cos_y"] - df_full["inter_cos_x"]
    # df_full['change_vis_intra'] = df_full.apply(lambda x: str(x.intra_cos_x) + " -> " + str(x.intra_cos_y), axis=1)
    # df_full['change_vis_inter'] = df_full.apply(lambda x: str(x.inter_cos_x) + " -> " + str(x.inter_cos_y), axis=1)
    pass
    # plot_umap_by('resolution', "upsampled", True)
    plot_umap_by('dataset', "base", True)

