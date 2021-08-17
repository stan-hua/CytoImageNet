from preprocessor import normalize as img_normalize
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalMaxPooling2D
import tensorflow as tf

from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.decomposition import PCA

import os
import cv2
import warnings
import datetime

import pandas as pd
import numpy as np
import scipy.stats as stats

import umap
import seaborn as sns
import matplotlib.pyplot as plt

# Plotting Settings
sns.set_style("dark")
plt.style.use('dark_background')

# PATHS
if "D:\\" in os.getcwd():
    annotations_dir = "M:/home/stan/cytoimagenet/annotations/"
    data_dir = 'M:/ferrero/stan_data/'
    evaluation_dir = "M:/home/stan/cytoimagenet/evaluation/"
    weights_dir = 'M:/home/stan/cytoimagenet/model/cytoimagenet-weights/'
else:
    annotations_dir = "/home/stan/cytoimagenet/annotations/"
    data_dir = '/ferrero/stan_data/'
    evaluation_dir = "/home/stan/cytoimagenet/evaluation/"
    weights_dir = '/home/stan/cytoimagenet/model/cytoimagenet-weights/'


# Only use CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Remove warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def load_metadata(dataset: str) -> pd.DataFrame:
    """Return metadata dataframe for <dataset>.
    """
    df_labels = None        # placeholder
    if dataset == "bbbc021":
        df_labels_1 = pd.read_csv(f"{data_dir}{dataset}/BBBC021_v1_image.csv")
        df_labels_2 = pd.read_csv(f"{data_dir}{dataset}/BBBC021_v1_moa.csv")
        df_labels = pd.merge(df_labels_1, df_labels_2, how="left",
                             left_on=["Image_Metadata_Compound",
                                      "Image_Metadata_Concentration"],
                             right_on=["compound", "concentration"])
        df_labels = df_labels[~df_labels.moa.isna()].reset_index(drop=True)
        df_labels['treatment'] = df_labels.apply(lambda x: f"{x.compound}-{x.concentration}", axis=1)
        df_labels = df_labels[~df_labels.treatment.str.contains("DMSO")].reset_index(drop=True)

    return df_labels


def load_model(weights='imagenet'):
    """Return EfficientNetB0 model. <weights> specify what weights to load into
    the model.
        - if weights == None, randomly initialize weights
        - elif weights == 'imagenet', load in weights from training on ImageNet
        - elif weights == 'cytoimagenet', load in weights from latest epoch
            of training on CytoImageNet.

    """
    if weights == "cytoimagenet":
        if False:
            weights = f"{weights_dir}random_init/efficientnetb0_from_random-epoch_10.h5"
            weights_notop = weights.replace(".h5", "-notop.h5")

            if not os.path.exists(weights_notop):
                model = tf.keras.models.Sequential()
                old_model = EfficientNetB0(weights=weights,
                                           input_shape=(224, 224, 3),
                                           pooling="max",
                                           classes=901)
                # Remove prediction layers
                old_model = tf.keras.Model(old_model.input, old_model.layers[-3].output)
                old_model.save_weights(weights_notop)

            # Load model weights with no top
            model = EfficientNetB0(weights=weights_notop,
                                   include_top=False,
                                   input_shape=(None, None, 3),
                                   pooling="max")
        else:
            # USING OLD 20 CLASS DATASET
            weights = f"{weights_dir}/efficientnetb0_from_random.h5"
            weights_notop = weights.replace(".h5", "-notop.h5")
            if not os.path.exists(weights_notop):
                old_model = tf.keras.Sequential([
                    EfficientNetB0(weights=None,
                                   input_shape=(224, 224, 3),
                                   pooling="max", include_top=False),
                    tf.keras.layers.Dense(20, 'softmax')
                ])
                old_model.load_weights(weights)
                # Remove prediction layers
                old_model = old_model.layers[:-1][0]
                old_model.save_weights(weights_notop)

            # Load model weights with no top
            model = EfficientNetB0(weights=weights_notop,
                                   include_top=False,
                                   input_shape=(None, None, 3),
                                   pooling="max")

    else:
        model = EfficientNetB0(weights=weights,
                               include_top=False,
                               input_shape=(None, None, 3),
                               pooling="max")
    return model


def extract_embeddings(concat=True, norm=False, weights="imagenet") -> pd.DataFrame:
    """Create activations for BBBC021.
        - If concat, extract embeddings for each channel. Else, average channels
            to produce 1 grayscale image, then extract embeddings.
        - If norm, then normalize image between [0 and 1] with respect to
            0.1th and 99.9th then normalize once again to [0, 255] by
            multiplying 255.
    """
    bbbc021_dir = f"{data_dir}bbbc021/"
    model = load_model(weights)

    # HELPER FUNCTION: Extract activations for each image
    def extract_activations(x):
        # Read channels
        imgs = []
        for channel in ["DAPI", "Tubulin", "Actin"]:
            img = cv2.imread(f"{bbbc021_dir}{x.Image_Metadata_Plate_DAPI}/{x[f'Image_FileName_{channel}']}", cv2.IMREAD_GRAYSCALE)
            if norm:    # Normalize between 0.1 and 99.9th percentile
                img = img_normalize(img) * 255
            # Convert grayscale to RGB
            imgs.append(np.stack([img] * 3, axis=-1))

        # If concat, extract embeddings from each image, then concatenate.
        if concat:
            return pd.Series(model.predict(np.array(imgs)).flatten())
        # Average channels
        img = np.array(imgs).mean(axis=0)
        return pd.Series(model.predict(np.expand_dims(img, axis=0)).flatten())

    df_metadata = load_metadata('bbbc021')
    activations = df_metadata.apply(extract_activations, axis=1)

    # Save extracted embeddings
    filename = f"{evaluation_dir}{weights}_embeddings/bbbc021_embeddings"
    if concat and norm:
        filename += f" ({weights}, concat, norm)"
    elif concat:
        filename += f" ({weights}, concat)"
    elif norm:
        filename += f" ({weights}, norm)"
    else:
        filename += f" ({weights})"

    activations.to_csv(f"{filename}.csv", index=False)
    return activations


def load_activations_bbbc021(concat=True, norm=False, weights="imagenet") -> pd.DataFrame:
    """Return dataframe of activations (samples, features), where rows
    correspond to rows in the metadata.
    """
    # Filename of embeddings
    filename = f"{evaluation_dir}{weights}_embeddings/bbbc021_embeddings"
    if concat and norm:
        filename += f" ({weights}, concat, norm)"
    elif concat:
        filename += f" ({weights}, concat)"
    elif norm:
        filename += f" ({weights}, norm)"
    else:
        filename += f" ({weights})"

        # Load embeddings if present
    if os.path.exists(f"{filename}.csv"):
        activations = pd.read_csv(f"{filename}.csv")
    else:
        # If not, extract embeddings
        activations = extract_embeddings(concat, norm, weights)
    return activations


def knn_classify(df_activations, unproc_labels: np.array, compounds: np.array,
                 k: int, metric='cosine', method="nsc", batch=None,
                 weights: str = 'imagenet', concat=True, norm=True):
    """Perform <k> Nearest Neighbors Classification of each sample, using
    activations in <df_activations>, and labels in <unproc_labels>.

    ==Precondition==:
        - <df_activations> and <unproc_labels> correspond. As a result their
            lengths should also be equal.
    ==Parameters==:
        - df_activations is array of (num_samples, num_features)
    """
    # Enforce precondition
    assert len(df_activations) == len(unproc_labels)
    if type(df_activations) == np.ndarray:
        df_activations = pd.DataFrame(df_activations)

    # Encode Labels
    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(unproc_labels)

    # Prediction/Evaluation Accumulator
    encoded_labels = np.unique(labels)
    correct_by_class = np.zeros(len(encoded_labels), dtype=np.uint32)
    total_by_class = np.zeros(len(encoded_labels), dtype=np.uint32)
    correct = 0.0
    total = 0.0

    # Fit KNN Classifier on the entire dataset
    knn_model = KNeighborsClassifier(n_neighbors=k, metric=metric)
    knn_model.fit(df_activations, labels)

    # Iterate through each sample. Match to existing samples.
    for i in range(len(df_activations)):
        print(f"Classifying MOA: {unproc_labels[i]}, Compound: {compounds[i]}")

        # Get k+1 neighbors of current sample, since 1st point is duplicate
        neigh_dist, neigh_ind = knn_model.kneighbors([df_activations.iloc[i]],
                                                     n_neighbors=k + 1)
        # Ignore duplicate point
        neigh_dist = neigh_dist[0][1:]
        neigh_ind = neigh_ind[0][1:]

        # Ignore Same Compound
        if method == 'nsc':
            # Filter only for other compounds. NOTE: Removes duplicate query
            num_same = np.sum(compounds[neigh_ind] == compounds[i])
            accum_num_same = num_same

            # If same compound found, get neighbors again
            while num_same > 0:
                neigh_dist, neigh_ind = knn_model.kneighbors(
                    [df_activations.iloc[i]], n_neighbors=k+1+accum_num_same)

                # Ignore duplicate point
                neigh_dist = neigh_dist[0][1:]
                neigh_ind = neigh_ind[0][1:]

                # Filter neighbors distances & indices
                idx_to_keep = np.argmax(compounds[neigh_ind] != compounds[i])
                neigh_dist = neigh_dist[idx_to_keep]
                neigh_ind = neigh_ind[idx_to_keep]
                # Update
                num_same = np.sum(compounds[neigh_ind] == compounds[i])
                accum_num_same += num_same

        neigh_labels = labels[neigh_ind]

        # If there is at least one non-unique value, take the mode, otherwise take the closest
        predicted = 0
        if not isinstance(neigh_labels, np.ndarray):    # if only one value
            predicted = neigh_labels
        elif len(np.unique(neigh_labels)) < len(neigh_labels):  # if more than one non-unique label, get mode
            predicted = stats.mode(neigh_labels)[0][0]
        else:
            smallest_ind = np.argmin(neigh_dist)
            predicted = neigh_labels[smallest_ind]

        # Check predictions
        if predicted == labels[i]:
            correct += 1.0
            correct_by_class[labels[i]] += 1
        total += 1.0
        total_by_class[labels[i]] += 1

    # Save Results
    df_results = pd.DataFrame()
    df_results['labels'] = np.unique(unproc_labels)
    df_results['correct_by_class'] = correct_by_class
    df_results['total_by_class'] = total_by_class
    df_results['accuracy_by_class'] = correct_by_class / total_by_class
    df_results['total_accuracy'] = correct / total

    df_results.to_csv(f"{evaluation_dir}/{weights}_results/bbbc021_kNN_results({weights}, {create_save_str(concat, norm)}, k-{k}).csv", index=False)


# Helper Functions:
def create_save_str(concat: bool, norm: bool):
    """Return string corresponding for saving options for <concat> and <norm>.
    """
    if concat and norm:
        save_str_params = "concat, norm"
    elif concat:
        save_str_params = "concat, no norm"
    elif norm:
        save_str_params = "merge, norm"
    else:
        save_str_params = "merge, no norm"
    return save_str_params


def get_results(concat, norm, weights):
    """Return Series containing kNN classification results averaged over all
    chosen k-values.
    """
    accum = []
    for k in [1, 5, 11, 25, 51]:
        df_results = pd.read_csv(f"{evaluation_dir}{weights}_results/bbbc021_kNN_results({weights}, {create_save_str(concat, norm)}, k-{k}).csv")
        accum.append(df_results.mean())
    df_return = pd.DataFrame(accum).mean().astype('object')
    df_return['to_grayscale'] = not concat
    df_return['normalized'] = norm
    return df_return


def get_all_results(weights):
    accum = []
    for concat in [True, False]:
        for norm in [True, False]:
            accum.append(get_results(concat, norm, weights))
    df_results = pd.DataFrame(accum)
    df_results['weights'] = weights
    df_results.to_csv(f"{evaluation_dir}bbbc021_aggregated_results({weights}).csv", index=False)


def timer(start, end):
    time_delta = (end - start)
    total_seconds = time_delta.total_seconds()
    minutes = total_seconds/60
    print("Finished in ", round(minutes, 2), " minutes!")
    return minutes


def umap_visualize(activations: np.array, labels: np.array, weights: str,
                   concat: bool, norm: bool, save: bool = True) -> None:
    """Plot 2D U-Map of extracted image embeddings for BBBC021.
    """
    # Extract UMap embeddings
    reducer = umap.UMAP(random_state=42)
    embeds = reducer.fit_transform(activations)

    # Create Plot
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
    plot_dir = "/home/stan/cytoimagenet/figures/"
    if save:
        if not os.path.isdir(f"{plot_dir}umap/"):
            os.mkdir(f"{plot_dir}umap/")
        plt.savefig(f"{plot_dir}umap/bbbc021_features({weights}, {create_save_str(concat, norm)}).png",
                    bbox_inches='tight', dpi=400)


if __name__ == "__main__" and "D:\\" not in os.getcwd():
    weights = 'cytoimagenet'

    # times = []
    # print(f"Beginning extraction of BBBC021 w/ concat and norm...")
    # start = datetime.datetime.now()
    # extract_embeddings(True, True, weights)
    # end = datetime.datetime.now()
    # times.append(timer(start, end))
    #
    # print(f"Beginning extraction of BBBC021 w/ only concat...")
    # start = datetime.datetime.now()
    # extract_embeddings(True, False, weights)
    # end = datetime.datetime.now()
    # times.append(timer(start, end))
    #
    # print(f"Beginning extraction of BBBC021 w/ only norm...")
    # start = datetime.datetime.now()
    # extract_embeddings(False, True, weights)
    # end = datetime.datetime.now()
    # times.append(timer(start, end))
    #
    # print(f"Beginning extraction of BBBC021 (only merged channels)...")
    # start = datetime.datetime.now()
    # extract_embeddings(False, True, weights)
    # end = datetime.datetime.now()
    # times.append(timer(start, end))
    #
    # with open(evaluation_dir+f"bbbc021_inference_times ({weights}).txt", "w+") as f:
    #     f.write("Feature extraction (concat, norm): ")
    #     f.write(str(times[0]) + " minutes\n")
    #     f.write("Feature extraction (concat): ")
    #     f.write(str(times[1]) + " minutes\n")
    #     f.write("Feature extraction (norm): ")
    #     f.write(str(times[2]) + " minutes\n")
    #     f.write("Feature extraction: ")
    #     f.write(str(times[3]) + " minutes\n")

    # else:
    # ==PARAMETERS==
    for weights in ['cytoimagenet', 'imagenet']:
        for concat in [True, False]:
            for norm in [True, False]:
                dir_name = "bbbc021"
                df_metadata = load_metadata(dir_name)
                df_activations = load_activations_bbbc021(concat, norm, weights)

                # Get list of MOA for each treatment
                labels = df_metadata.groupby(by=['treatment']).sample(n=1).moa.to_numpy()
                # Get list of compounds for each treatment
                compounds = df_metadata.groupby(by=['treatment']).sample(n=1).compound.to_numpy()

                # Add treatment label to activations
                df_activations['treatment'] = df_metadata['treatment']
                # Get mean feature vectors for each treatment
                treatment_activations = df_activations.groupby(by=['treatment']).mean()

                # # Normalize features between [0, 1]
                # proc_treatment_activations = preprocessing.normalize(treatment_activations)
                #
                # # Reduce Dimensionality of Treatment Activations
                # pca = PCA()
                # proc_treatment_activations = pca.fit_transform(proc_treatment_activations)
                # # Select only >= 99% cumulative percent variance
                # cum_percent_variance = pca.explained_variance_ratio_.cumsum()
                # num_pc = np.where(cum_percent_variance >= 0.99)[0][0]
                # proc_treatment_activations = proc_treatment_activations[:, : num_pc+1]

                k_values = [1, 5, 11, 25, 51]
                for k in k_values:
                    knn_classify(treatment_activations, labels, compounds, k,
                                 concat=concat, norm=norm, weights=weights)

                # Create UMap visualization of features
                umap_visualize(treatment_activations, labels=labels, weights=weights,
                               concat=concat, norm=norm, save=True)

        # Get final results
        get_all_results(weights)
