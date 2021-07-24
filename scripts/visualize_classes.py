from prepare_dataset import check_exists
from feature_extraction import intra_cos_sims, inter_cos_sims, get_summary_similarities
from typing import Union

import numpy as np
import pandas as pd
from scipy.stats import iqr
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import random
import os
import glob

from sklearn.cluster import DBSCAN
import umap
import umap.plot


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
    df.apply(check_exists, axis=1)

    imgs = []
    for i in df.index:
        try:
            im = Image.open(df.loc[i, "path"] + "/" + df.loc[i, "filename"])
            imgs.append(np.array(im.resize((224, 224))))
        except:
            pass
    return imgs


def gridplot_images(imgs: list, label:str):
    # Randomly sample 100
    try:
        img_samples = random.sample(imgs, 25)
    except:
        return False

    # settings
    h, w = 5, 5        # for raster image
    nrows, ncols = 5, 5  # array of sub-plots
    figsize = [6, 8]     # figure size, inches

    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # plot simple raster image on each sub-plot
    n = 0
    for i, axi in enumerate(ax.flat):
        axi.imshow(img_samples[n], cmap="gray")
        axi.set_axis_off()
        n += 1
    fig.suptitle(label)
    plt.savefig(f"{plot_dir}class_grid_show/{label}_grid.png")


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


def create_umap(labels: Union[str, list],
                directory: str = "imagenet-activations/"):
    """Return tuple of UMAP 2D embeddings and labels for each row.

    ==Parameters==:
        directory: specifies whether to use activations from randomly initiated
            AlexNet embeddings, or ImageNet pretrained EfficientNetB0.
            - either "imagenet-activations/" or "random_model-activations/"
        name: save figure as <name>.png
    """
    all_activations = []
    if isinstance(labels, str):
        activations = pd.read_csv(f"{model_dir}{directory}/{labels}_activations.csv")
        # Reference for row to label
        label_handle = [labels] * len(activations)
    else:
        # Accumulate activations & label handle
        activations = []
        label_handle = []

        for label in labels:
            temp = pd.read_csv(f"{model_dir}{directory}/{label}_activations.csv")
            activations.append(temp)
            label_handle.extend([label] * len(temp))

        activations = pd.concat(activations)

    # Find 2D U-Map Embeddings
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(activations)

    return embedding, np.array(label_handle)


def plot_umap(embeds: np.array, labels: list, name: str = "", save: bool = False):
    """Plot 2D U-Map of extracted image embeddings for <labels>.

    ==Parameters==:
        name: save figure as <name>.png
    """
    plt.figure()
    ax = sns.scatterplot(x=embeds[:, 0], y=embeds[:, 1],
                         hue=labels,
                         legend="full",
                         alpha=1,
                         palette="tab20",
                         s=2,
                         linewidth=0)
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


def from_label_to_paths(label: str, kind: str):
    """Return all image paths for <label> under <kind> preprocessing.

    ==Preconditions==:
        - kind is either 'base' (no modification) or 'upsampled'
    """
    if kind == "base":
        df = pd.read_csv(f"{annotations_dir}classes/{label}.csv")
    else:
        df = pd.read_csv(f"{annotations_dir}classes/{kind}/{label}.csv")
    return df.apply(lambda x: x.path + "/" + x.filename, axis=1).tolist()


if __name__ == "__main__" and "D:\\" not in os.getcwd():
    next_chosen = ['human', 'nucleus', 'cell membrane',
                   'white blood cell', 'kinase',
                   'wildtype', 'difficult',
                   'nematode', 'yeast', 'bacteria',
                   ]

    # for label in next_chosen:
    #     imgs = load_images_from_label(label)
    #     if gridplot_images(imgs, label) is None:
    #         print("Success! for " + label)
    #     else:
    #         print("No images for " + label)
    # df_base = get_summary_similarities(embeds, labels)
    # df_base.to_csv(model_dir + "similarity/base.csv", index=False)

    for kind in ["base", "upsampled"]:
        # Get Embeddings
        embeds, labels = create_umap(next_chosen, directory=f"imagenet-activations/{kind}/")
        # Plot U-Map labeled by category labels
        print(embeds.shape, len(labels))
        plot_umap(embeds, labels, name=kind, save=True)

        # Convert to dataframe
        df_embed = pd.DataFrame(embeds)
        df_embed["labels"] = labels
        df_embed.to_csv(model_dir + f'imagenet-activations/{kind}_embeddings.csv', index=False)

        # Get image paths to corresponding activations
        full_path = []
        for label in np.unique(labels):
            full_path.extend(from_label_to_paths(label, kind))
        len(full_path)
        df_embed['full_path'] = full_path

        # Density-based Clustering
        cluster_labels = cluster_by_density(df_embed)

        # Gridplot 25 images from each cluster
        for cluster in np.unique(cluster_labels):
            embed_cluster = df_embed.loc[cluster_labels == cluster]
            if len(embed_cluster) > 25:
                embed_cluster = embed_cluster.sample(n=25)
            imgs = [Image.open(path) for path in embed_cluster.full_path.tolist()]
            print(f"{kind} cluster {cluster} ", np.array(imgs).flatten().mean())
            gridplot_images(imgs, f"{kind} cluster_{cluster}")

        # Plot U-Map labeled by cluster assignment
        plot_umap(df_embed.iloc[:, :2].to_numpy(), cluster_labels, name=f"{kind}_clustered")

elif "D:\\" in os.getcwd():
    # df_base = pd.read_csv(model_dir + "similarity/base.csv")
    # df_up = pd.read_csv(model_dir + "similarity/upsampled.csv")
    # df_full = pd.merge(df_base, df_up, on="label")
    # df_full['change_intra_cos'] = df_full["intra_cos_y"] - df_full["intra_cos_x"]
    # df_full['change_inter_cos'] = df_full["inter_cos_y"] - df_full["inter_cos_x"]
    # df_full['change_vis_intra'] = df_full.apply(lambda x: str(x.intra_cos_x) + " -> " + str(x.intra_cos_y), axis=1)
    # df_full['change_vis_inter'] = df_full.apply(lambda x: str(x.inter_cos_x) + " -> " + str(x.inter_cos_y), axis=1)

    if True:
        df_embed = pd.read_csv(model_dir + 'imagenet-activations/base_embeddings.csv')
        datasets_from = []
        for label in df_embed.labels.unique():
            df = pd.read_csv(f"{annotations_dir}classes//{label}.csv")
            datasets_from.extend(df.name.tolist())
    else:
        df_embed = pd.read_csv(model_dir + 'imagenet-activations/upsampled_embeddings.csv')
        datasets_from = []
        for label in df_embed.labels.unique():
            df = pd.read_csv(f"{annotations_dir}classes/upsampled/{label}.csv")
            datasets_from.extend(df.name.tolist())

    # plot_umap(df_embed.iloc[:, :2].to_numpy(), datasets_from, name="", save=False)

    plt.figure()
    ax = sns.scatterplot(x=df_embed.iloc[:, :2].to_numpy()[:, 0], y=df_embed.iloc[:, :2].to_numpy()[:, 1],
                         hue=datasets_from,
                         legend='full',
                         alpha=1,
                         palette="gist_rainbow",
                         s=2,
                         linewidth=0)
    plt.tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)
    legend_handles = ax.get_legend_handles_labels()
    ax.get_legend().remove()

    plt.figure(figsize=(4.15, 7.3))
    pylab.figlegend(*ax.get_legend_handles_labels(), loc = 'upper left')
