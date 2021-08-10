from preprocessor import normalize
from tensorflow.keras.applications import EfficientNetB0

import pandas as pd
import numpy as np

import os
import cv2

import csv
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import scipy.stats as stats
import warnings

# PATHS
if "D:\\"  in os.getcwd():
    annotations_dir = "M:/home/stan/cytoimagenet/annotations/"
    data_dir = 'M:/ferrero/stan_data/'
    evaluation_dir = "M:/home/stan/cytoimagenet/evaluation/"
    weights_dir = 'M:/home/stan/cytoimagenet/model/cytoimagenet-weights/random_init/'
else:
    annotations_dir = "/home/stan/cytoimagenet/annotations/"
    data_dir = '/ferrero/stan_data/'
    evaluation_dir = "/home/stan/cytoimagenet/evaluation/"
    weights_dir = '/home/stan/cytoimagenet/model/cytoimagenet-weights/random_init/'


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
        weights = f"{weights_dir}/efficientnetb0_from_random-epoch_10.h5"
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
                img = normalize(img) * 255
            # Convert grayscale to RGB
            imgs.append(np.stack([img] * 3, axis=-1))

        # If concat, extract embeddings from each image, then concatenate.
        if concat:
            return pd.Series(model.predict(np.array(imgs)).flatten())
        # Average channels
        img = np.stack(imgs, axis=-1).mean()
        return pd.Series(model.predict(np.expand_dims(img, axis=0)).flatten())

    df_metadata = load_metadata('bbbc021')
    activations = df_metadata.apply(extract_activations, axis=1)

    print(activations)

    # Save extracted embeddings
    filename = f"{evaluation_dir}bbbc021_embeddings"
    if concat and norm:
        filename += " (concat, norm)"
    elif concat:
        filename += " (concat)"
    elif norm:
        filename += " (norm)"

    activations.to_csv(f"{filename}.csv", index=False)
    return activations


def load_activations_bbbc021(concat=True, norm=False, weights="imagenet") -> pd.DataFrame:
    """Return dataframe of activations (samples, features), where rows
    correspond to rows in the metadata.
    """
    # Filename of embeddings
    filename = f"{evaluation_dir}bbbc021_embeddings"
    if concat and norm:
        filename += " (concat, norm)"
    elif concat:
        filename += " (concat)"
    elif norm:
        filename += " (norm)"

    # Load embeddings if present
    if os.path.exists(f"{filename}.csv"):
        activations = pd.read_csv(f"{filename}.csv")
    else:
        # If not, extract embeddings
        activations = extract_embeddings(concat, norm, weights)
    return activations


def knn_classify(df_activations: pd.DataFrame, unproc_labels, k,
                 metric='euclidean', weights: str = 'imagenet'):
    """Perform <k> Nearest Neighbors Classification of each sample, using
    activations in <df_activations>, and labels in <unproc_labels>.

    ==Precondition==:
        - <df_activations> and <unproc_labels> correspond. As a result their
            lengths should also be equal.

    """
    # Enforce precondition
    assert len(df_activations) == len(unproc_labels)

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
        print(f"Classifying {labels[i]} of {len(df_activations)}")

        # Get k+1 neighbors of current sample, since 1st point is duplicate
        neigh_dist, neigh_ind = knn_model.kneighbors([df_activations.iloc[i]],
                                                     n_neighbors=k + 1)
        # Ignore duplicate point
        neigh_dist = neigh_dist[0][1:]
        neigh_ind = neigh_ind[0][1:]
        neigh_labels = labels[neigh_ind]

        # If there is at least one non-unique value, take the mode, otherwise take the closest
        predicted = 0
        if len(np.unique(neigh_labels)) < len(neigh_labels):
            predicted = stats.mode(neigh_labels)[0][0]
        else:
            smallest_ind = np.argmin(neigh_dist)
            predicted = neigh_labels[smallest_ind]
        if predicted == labels[i]:
            correct += 1.0
            correct_by_class[labels[i]] += 1
        total += 1.0
        total_by_class[labels[i]] += 1

    # Save Results
    df_results = pd.DataFrame()
    df_results['labels'] = np.unique(labels)
    df_results['correct_by_class'] = correct_by_class
    df_results['total_by_class'] = total_by_class
    df_results['accuracy_by_class'] = np.float(correct_by_class/total_by_class)
    df_results['total_accuracy'] = correct / total
    df_results.to_csv(f"{evaluation_dir}/bbbc021_results ({weights}).csv")



if __name__ == "__main__" and "D:\\" not in os.getcwd():
    dir_name = "bbbc021"
    df_metadata = load_metadata(dir_name)
    df_activations = load_activations_bbbc021()
    # Add treatment label to activations
    df_activations['treatment'] = df_metadata['treatment']
    # TODO: Get list of treatment to MOA
    labels = df_metadata.groupby(by=['treatment']).sample(n=1).moa.tolist()

    treatment_activations = df_activations.groupby(by=['treatment']).mean()
    treatments = treatment_activations.index.tolist()

    # TODO: Normalize features between [0, 1]
    # TODO: Reduce Dimensionality of Treatment Activations



    basename = './best_features/'
    outname = './best_results/'

    k_values = [1, 5, 11, 25, 50]
