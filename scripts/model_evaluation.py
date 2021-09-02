from tensorflow.keras.applications import EfficientNetB0
import tensorflow as tf

from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

import os
import glob
import cv2
import warnings
import datetime
from typing import List, Optional

import pandas as pd
import numpy as np
import scipy.stats as stats
from math import ceil

import umap
import seaborn as sns
import matplotlib.pyplot as plt

# Add scripts subdirectory to PATH
import sys
sys.path.append("./data_processing")
from preprocessor import normalize as img_normalize

# Plotting Settings
sns.set_style("white")
plt.style.use('seaborn-white')

# PATHS
if "D:\\" in os.getcwd():
    annotations_dir = "M:/home/stan/cytoimagenet/annotations/"
    data_dir = 'M:/ferrero/stan_data/'
    evaluation_dir = "M:/home/stan/cytoimagenet/evaluation/"
    weights_dir = 'M:/home/stan/cytoimagenet/model/cytoimagenet-weights/'
    plot_dir = "M:/home/stan/cytoimagenet/figures/"
else:
    annotations_dir = "/home/stan/cytoimagenet/annotations/"
    data_dir = '/ferrero/stan_data/'
    evaluation_dir = "/home/stan/cytoimagenet/evaluation/"
    weights_dir = '/home/stan/cytoimagenet/model/cytoimagenet-weights/'
    plot_dir = "/home/stan/cytoimagenet/figures/"


# Only use CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Remove warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def load_model(weights='cytoimagenet',
               weights_filename='efficientnetb0_from_random-epoch_16.h5',
               init='random', overwrite=False, dset_=None):
    """Return EfficientNetB0 model. <weights> specify what weights to load into
    the model.
        - if weights == None, randomly initialize weights
        - elif weights == 'imagenet', load in weights from training on ImageNet
        - elif weights == 'cytoimagenet', load in weights from latest epoch
            of training on CytoImageNet.

    """
    if weights == "cytoimagenet":
        # Specify number of classes based on dset
        if dset_ == 'toy_20':
            num_classes = 20
        elif dset_ == 'toy_50':
            num_classes = 50
        else:
            num_classes = 894

        # Load weights
        weights_str = f"{weights_dir}/{init}_init/{dset_}/{weights_filename}"
        weights_notop = weights_str.replace(".h5", "-notop.h5")

        # Save notop weights if they don't exist
        if not os.path.exists(weights_notop) or overwrite:
            model_withtop = EfficientNetB0(weights=None,
                                           input_shape=(224, 224, 3),
                                           classes=num_classes)
            model_withtop.load_weights(weights_str)

            # Save weights without prediction layers
            model = tf.keras.Model(model_withtop.input, model_withtop.layers[-3].output)
            model.save_weights(weights_notop)

        # Load model weights with no top
        model = EfficientNetB0(weights=weights_notop,
                               include_top=False,
                               input_shape=(None, None, 3),
                               pooling="avg")
    elif weights is None:
        weights_str = f'/home/stan/cytoimagenet/model/random_efficientnetb0-notop.h5'
        if not os.path.exists(weights_str):     # if random weights doesn't exist yet
            model = EfficientNetB0(weights=None,
                                   include_top=False,
                                   input_shape=(None, None, 3),
                                   pooling="avg")
            model.save_weights(weights_str)
        model = EfficientNetB0(weights=weights_str,
                               include_top=False,
                               input_shape=(None, None, 3),
                               pooling="avg")
    else:   # if ImageNet or random
        model = EfficientNetB0(weights=weights,
                               include_top=False,
                               input_shape=(None, None, 3),
                               pooling="avg")
    return model


# HELPER FUNCTIONS:
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


def check_weights_str(weights):
    if weights is not None:     # None -> 'random'
        weights_str = weights
    else:
        weights_str = 'random'
    return weights_str


def timer(start, end):
    time_delta = (end - start)
    total_seconds = time_delta.total_seconds()
    minutes = total_seconds/60
    print("Finished in ", round(minutes, 2), " minutes!")
    return minutes


class ImageGenerator:
    """ImageGenerator Class. Used to return multi-channel images.

    ==Attributes==:
        path_gens: list of lists containing absolute paths
                    to each channel image.
        concat: Boolean. If True, concatenate features for channels. If False,
                    merge image channels into 1 image then extract features.
        norm: Boolean. If True, normalize each channel image with respect to 0.1
                    and 99.9th percentile pixel intensity.
        labels: OPTIONAL. If specified, generator will yield label corresponding
                    to image.
    """
    def __init__(self, path_gens: list, concat: bool, norm: bool,
                 labels: Optional[list] = None):
        self.path_gens = path_gens
        self.concat = concat
        self.norm = norm
        self.labels = labels

    def get_image(self, paths: list):
        """Returns channel image/s after image operations specified in
        class attributes.
        """
        imgs = []
        for path in paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            # Normalize between 0.1 and 99.9th percentile
            if self.norm:
                img = img_normalize(img) * 255

            # Convert grayscale to RGB
            imgs.append(np.stack([img] * 3, axis=-1))

        # If concat, return array of images
        if self.concat:
            return np.array(imgs)

        # Else, average over channel images to create single grayscale image
        img = np.array(imgs).mean(axis=0)
        return np.expand_dims(img, axis=0)

    def __iter__(self):
        for i in range(len(self)):
            # Get all absolute paths to channel images
            paths = [gen[i] for gen in self.path_gens]

            if self.labels is not None:     # (img, label)
                yield self.get_image(paths), self.labels[i]
            else:      # img
                yield self.get_image(paths)

    def __len__(self):
        return len(self.path_gens[0])


class ValidationProcedure:
    """Abstract ValidationProcedure Class. Meant to serve as a base for
    subclasses BBBC021Protocol, COOS7Validation, and CyCLOPSValidation.
    Do NOT instantiate this class.

    Generally, the same procedure is taken for all evaluation sets:
        1. Create Image Generators for each channel.
        2. Extract embeddings based on specific preprocessing method.
        3. Classify using k - Nearest Neighbors.
        4. Save results.

    ==Attributes==:
        metadata: pandas DataFrame containing paths to channel images
        data_dir: directory location of dataset
        name: name of dataset (convention)
        num_channels: number of channels for each image
        path_gens: list of <num_channels> path iterators one for each channel
        len_dataset: number of <num_channel> unique images in dataset
        k_values: tuple of k values to test in kNN
    """
    def __init__(self, dset):
        self.metadata = self.load_metadata()
        self.data_dir = ''
        self.name = ''
        self.num_channels = 3
        self.path_gens = self.create_path_iterators()
        self.len_dataset = len(self.path_gens[0])
        self.k_values = (1,)
        self.dset = dset

    def load_metadata(self) -> pd.DataFrame:
        """ABSTRACT METHOD. To be implemented in subclass.
        Return metadata dataframe for BBBC021.
        """
        raise NotImplementedError()

    def create_path_iterators(self) -> List[List[str]]:
        """ABSTRACT METHOD. To be implemented in subclass.
        Returns list containing list of absolute paths for each channel."""
        raise NotImplementedError()

    def extract_embeddings(self, concat=True, norm=False, weights="imagenet",
                           overwrite=False) -> pd.DataFrame:
        """Extract activations/embeddings for evaluation set.
            - If concat, extract embeddings for each channel. Else, average channels
                to produce 1 grayscale image, then extract embeddings.
            - If norm, then normalize image between [0 and 1] with respect to
                0.1th and 99.9th then normalize once again to [0, 255] by
                multiplying 255.
        """
        # Load model
        model = load_model(weights, overwrite=overwrite, dset_=self.dset)

        # Create efficient data generators
        test_generator = ImageGenerator(self.path_gens, concat, norm)
        ds_test = tf.data.Dataset.from_generator(lambda: test_generator, output_types=(tf.float32))

        # Prefetch data in batches
        batch_size = 1
        # ds_test = ds_test.batch(batch_size)
        ds_test = ds_test.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        steps_to_predict = ceil(len(test_generator) / batch_size)
        # Extract embeddings
        accum_activations = model.predict(ds_test, verbose=2,
                                          steps=steps_to_predict,
                                          use_multiprocessing=True, workers=32)

        # Concatenate <num_channels> consecutive feature vectors that correspond to 1 image
        if concat:
            proc_activations = []
            for i in range(self.num_channels, len(accum_activations)+1,
                           self.num_channels):
                proc_activations.append(accum_activations[i-self.num_channels:i].flatten())
            activations = pd.DataFrame(proc_activations)
        else:
            activations = pd.DataFrame(accum_activations)

        # Save extracted embeddings
        weights_str = check_weights_str(weights)
        filename = f"{evaluation_dir}{weights_str}_embeddings/{self.name}_embeddings"
        filename += f" ({weights_str}, {create_save_str(concat, norm)})"

        activations.to_hdf(f"{filename}.h5", key='embed', index=False)
        return activations

    def load_activations(self, concat=True, norm=False, weights="imagenet",
                         overwrite=False) -> pd.DataFrame:
        """Return dataframe of activations (samples, features), where rows
        correspond to rows in the metadata. Activations are centered and
        standardized.

        If <overwrite>, extract embeddings again.
        """
        # Filename of embeddings
        weights_str = check_weights_str(weights)
        filename = f"{evaluation_dir}{weights_str}_embeddings/{self.name}_embeddings"
        suffix = f" ({weights_str}, {create_save_str(concat, norm)})"
        # Load embeddings if present
        if os.path.exists(f"{filename}{suffix}.h5") and not overwrite:
            activations = pd.read_hdf(f"{filename}{suffix}.h5", 'embed')
        else:
            # If not, extract embeddings
            print(f"Beginning extraction of {self.name.upper()} w/{suffix}...")
            start = datetime.datetime.now()
            activations = self.extract_embeddings(concat, norm, weights, overwrite=overwrite)
            end = datetime.datetime.now()
            total = timer(start, end)

            # Log time taken
            if not os.path.exists(evaluation_dir + f"{self.name}_inference_times({weights_str}).csv"):
                df_time = pd.DataFrame({'to_grayscale': not concat, 'normalize': norm, 'minutes': total}, index=[0])
                df_time.to_csv(evaluation_dir + f"{self.name}_inference_times({weights_str}).csv", index=False)
            else:
                df_time = pd.read_csv(evaluation_dir + f"{self.name}_inference_times({weights_str}).csv")
                # Remove past record
                df_time = df_time[(df_time.to_grayscale == concat) | (df_time.normalize != norm)]
                # Add existing time
                df_curr = pd.DataFrame({'to_grayscale': not concat, 'normalize': norm, 'minutes': total}, index=[0])
                df_time = pd.concat([df_time, df_curr])
                df_time.to_csv(evaluation_dir + f"{self.name}_inference_times({weights_str}).csv", index=False)

        # Subtract Mean and Divide by Standard Deviation
        scaler = preprocessing.StandardScaler()
        transformed_activations = scaler.fit_transform(activations)

        return pd.DataFrame(transformed_activations)

    # Classifier
    def knn_classify(self, df_activations, unproc_labels: np.array,
                     k: int, metric='euclidean',
                     weights: str = 'imagenet', concat=True, norm=True):
        """Perform <k> Nearest Neighbors Classification of each sample, using
        activations in <df_activations>, and labels in <unproc_labels>.

        kNN code is modified from Alex Lu's COOS-7 paper.

        ==Precondition==:
            - <df_activations> and <unproc_labels> are of equal lengths.
            - features in <df_activations> are centered and standardized by
                mean and std.
        ==Parameters==:
            - df_activations is array of (num_samples, num_features)
        """
        # Enforce precondition
        assert len(df_activations) == len(unproc_labels)
        # Convert from numpy array to pandas dataframe
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
            # Get k+1 neighbors of current sample, since 1st point is duplicate
            neigh_dist, neigh_ind = knn_model.kneighbors([df_activations.iloc[i]],
                                                         n_neighbors=k + 1)
            # Ignore duplicate point
            neigh_dist = neigh_dist[0][1:]
            neigh_ind = neigh_ind[0][1:]

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

        weights_str = check_weights_str(weights)
        df_results.to_csv(f"{evaluation_dir}/{weights_str}_results/{self.name}_kNN_results({weights_str}, {create_save_str(concat, norm)}, k-{k}).csv", index=False)

    # MAIN FUNCTION
    def evaluate(self, weights, concat, norm):
        """ABSTRACT METHOD. To be implemented in subclass.
        Main function to carry out evaluation protocol."""
        raise NotImplementedError()

    # Collect Results
    def get_results(self, concat, norm, weights):
        """Return Series containing kNN classification results averaged over all
        chosen k-values.
        """
        weights_str = check_weights_str(weights)       # converts None -> 'random'

        accum = []
        for k in self.k_values:
            df_results = pd.read_csv(f"{evaluation_dir}{weights_str}_results/{self.name}_kNN_results({weights_str}, {create_save_str(concat, norm)}, k-{k}).csv")
            accum.append(df_results.mean())
        df_return = pd.DataFrame(accum).mean().astype('object')
        # Get 95% confidence interval on accuracy
        acc = df_return['total_accuracy']
        df_return['ci'] = 1.96 * np.sqrt((acc * (1 - acc)) / self.len_dataset)
        df_return['to_grayscale'] = not concat
        df_return['normalized'] = norm
        return df_return

    def get_all_results(self, weights):
        """Aggregates results for <weights> and saves it into a csv file."""
        weights_str = check_weights_str(weights)       # converts None -> 'random'
        accum = []
        for concat in [True, False]:
            for norm in [True, False]:
                accum.append(self.get_results(concat, norm, weights))
        df_results = pd.DataFrame(accum)
        df_results['weights'] = weights_str
        df_results.to_csv(f"{evaluation_dir}{self.name}_aggregated_results({weights_str}).csv", index=False)

    # Plots Aggregated Results
    def plot_all_results(self, save=True, by='method'):
        """Creates bar plots for accuracy comparing random, ImageNet and
        CytoImageNet features."""
        all_weights = ['imagenet', 'cytoimagenet', 'random']

        # Get accumulated results
        accum_df = []
        for i in range(len(all_weights)):
            curr_df = pd.read_csv(f"{evaluation_dir}{self.name}_aggregated_results({all_weights[i]}).csv")
            if curr_df.ci.isna().sum() > 0 and self.name == 'bbbc021':
                curr_df['ci'] = curr_df.apply(lambda x: 1.96 * np.sqrt((x.total_accuracy * (1 - x.total_accuracy)) / 103), axis=1)
            curr_df['preproc_method'] = curr_df.apply(lambda x: create_save_str(not x.to_grayscale, x.normalized),
                                                      axis=1)
            accum_df.append(curr_df)
        df = pd.concat(accum_df, ignore_index=True)

        # Plotting by preprocessing method, or by weights
        if by == 'method':
            fig, axs = plt.subplots(1, 4, sharey=True)
            axs = axs.ravel()
            methods = df.preproc_method.unique()
            for i in range(len(methods)):
                curr_df = df[df.preproc_method == methods[i]]
                # Plot
                sns.scatterplot(x="weights", y="total_accuracy", hue='weights',
                                data=curr_df, ax=axs[i], palette='dark')
                # Error Bars
                axs[i].errorbar(x=curr_df['weights'],y=curr_df['total_accuracy'],
                                yerr=curr_df['ci'], fmt='none', c='black', zorder=0,
                                alpha=0.5)

                axs[i].set_title(methods[i].capitalize())
                axs[i].set_xlabel('')
                axs[i].tick_params(axis='x', rotation=90)
                axs[i].set_ylim([0, 1.0])
                axs[i].get_legend().remove()
                if i > 0:
                    axs[i].set_ylabel('')
                else:
                    axs[i].set_ylabel('Accuracy')
        else:
            fig, axs = plt.subplots(1, 3, sharey=True)
            axs = axs.ravel()
            for i in range(len(all_weights)):
                curr_df = df[df.weights == all_weights[i]]
                if curr_df.ci.isna().sum() > 0 and self.name == 'bbbc021':
                    curr_df['ci'] = curr_df.apply(lambda x: 1.96 * np.sqrt((x.total_accuracy * (1 - x.total_accuracy)) / 103), axis=1)
                curr_df['preproc_method'] = curr_df.apply(lambda x: create_save_str(not x.to_grayscale, x.normalized),
                                                          axis=1)
                accum_df.append(curr_df)

                # Plot
                sns.pointplot(x="preproc_method", y="total_accuracy",
                              hue="preproc_method",
                              data=curr_df, ax=axs[i])
                # Error Bars
                axs[i].errorbar(x=curr_df['preproc_method'],y=curr_df['total_accuracy'],
                                yerr=curr_df['ci'], fmt='none', c='white', zorder=0,
                                alpha=0.5)

                axs[i].set_title(all_weights[i].capitalize())
                axs[i].set_xlabel('')
                axs[i].tick_params(axis='x', rotation=45)
                axs[i].set_ylim([0, 1.0])
                axs[i].get_legend().remove()
                if i > 0:
                    axs[i].set_ylabel('')
                else:
                    axs[i].set_ylabel('Accuracy')

        if save:
            plt.tight_layout()
            plt.savefig(f"{plot_dir}evaluation/{self.name}_results({by}).png")

    # UMAP Visualization
    def umap_visualize(self, activations: np.array, labels: np.array, weights: str,
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
            if not os.path.isdir(f"{plot_dir}umap/{self.name}"):
                os.mkdir(f"{plot_dir}umap/{self.name}")
            weights_str = check_weights_str(weights)       # converts None -> 'random'
            plt.savefig(f"{plot_dir}umap/{self.name}/{self.name}_features({weights_str}, {create_save_str(concat, norm)}).png",
                        bbox_inches='tight', dpi=400)


class BBBC021Protocol(ValidationProcedure):
    """BBBC021 Evaluation Protocol. 1-nearest neighbors classification of
    mechanism-of-action (MOA). Reports NSC Accuracy.
        NOTE: Aggregates ~2000 feature vectors by 103 treatments to get 103
                feature vectors.
        NOTE: Time for feature extraction is recorded.
    """
    def __init__(self, dset):
        self.data_dir = f"{data_dir}bbbc021"
        self.metadata = self.load_metadata()
        self.name = 'bbbc021'
        self.num_channels = 3
        self.path_gens = self.create_path_iterators()
        self.len_dataset = len(self.path_gens[0])
        self.k_values = (1,)
        self.dset = dset

    def load_metadata(self) -> pd.DataFrame:
        """Return metadata dataframe for BBBC021.
        """
        # Get metadata
        df_temp_1 = pd.read_csv(f"{self.data_dir}/BBBC021_v1_image.csv")
        df_temp_2 = pd.read_csv(f"{self.data_dir}/BBBC021_v1_moa.csv")
        df_metadata = pd.merge(df_temp_1, df_temp_2, how="left", left_on=["Image_Metadata_Compound",
                                                                          "Image_Metadata_Concentration"], right_on=["compound", "concentration"])
        df_metadata = df_metadata[~df_metadata.moa.isna()].reset_index(drop=True)
        df_metadata['treatment'] = df_metadata.apply(lambda x: f"{x.compound}-{x.concentration}", axis=1)
        # Exclude DMSO treatment
        df_metadata = df_metadata[~df_metadata.treatment.str.contains("DMSO")].reset_index(drop=True)

        return df_metadata

    def create_path_iterators(self) -> List[List[str]]:
        """Return list containing list of absolute paths for each channel."""
        # Create absolute path iterators
        path_gens = []
        for channel in ["DAPI", "Tubulin", "Actin"]:
            path_gens.append(self.metadata.apply(lambda x: f"{self.data_dir}/{x.Image_Metadata_Plate_DAPI}/{x[f'Image_FileName_{channel}']}",
                axis=1).tolist())
        return path_gens

    # Modified kNN for Not-Same-Compound (NSC) MOA classification
    def knn_classify(self, df_activations, unproc_labels: np.array, compounds: np.array,
                     k: int, metric='cosine', method="nsc",
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
            # print(f"Classifying MOA: {unproc_labels[i]}, Compound: {compounds[i]}")

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

        weights_str = check_weights_str(weights)
        df_results.to_csv(f"{evaluation_dir}/{weights_str}_results/{self.name}_kNN_results({weights_str}, {create_save_str(concat, norm)}, k-{k}).csv", index=False)

    # MAIN FUNCTION
    def evaluate(self, weights, concat, norm, overwrite=False):
        """Main function to carry out evaluation protocol.
        If overwrite, ignore existing embeddings and extract.
        """
        df_activations = self.load_activations(concat, norm, weights, overwrite=overwrite)

        # Get list of MOA for each treatment
        labels = self.metadata.groupby(by=['treatment']).sample(n=1).moa.to_numpy()
        # Get list of compounds for each treatment
        compounds = self.metadata.groupby(by=['treatment']).sample(n=1).compound.to_numpy()

        # Add treatment label to activations
        df_activations['treatment'] = self.metadata['treatment']
        # Get mean feature vectors for each treatment
        treatment_activations = df_activations.groupby(by=['treatment']).mean()

        # kNN Classifier
        for k in self.k_values:
            self.knn_classify(treatment_activations, labels,
                              compounds, k, concat=concat,
                              norm=norm, weights=weights)

        # Create UMAP visualization of features
        self.umap_visualize(treatment_activations, labels=labels, weights=weights,
                       concat=concat, norm=norm, save=True)


class COOS7Validation(ValidationProcedure):
    """Cells Out Of Sample Dataset (COOS-7) Dataset Evaluation.
    Specify choice of test set:
        - test1, test2, test3, test4

    ==Parameters==:
        test: one of 'test1', test2', 'test3', 'test4'. Specifies which test set
                to use.
    """
    def __init__(self, dset, test):
        self.data_dir = f'/neuhaus/alexlu/datasets/IMAGE_DATASETS/COOS-MOUSE_david-andrews/CURATED-COOS7/{test}'
        self.metadata = self.load_metadata()
        self.test_set = test
        self.name = f'coos7_{test}'
        self.num_channels = 3
        self.path_gens = self.create_path_iterators()
        self.len_dataset = len(self.path_gens[0])
        self.k_values = (11,)
        self.dset = dset

    def load_metadata(self) -> pd.DataFrame:
        """Return metadata dataframe for COOS-7.
        """
        files = glob.glob(os.sep.join([self.data_dir, '*']))
        # Get labels
        labels = [file.split(self.data_dir + os.sep)[-1] for file in files]

        accum_df = []
        for i in range(len(labels)):
            # Get all files for label
            label_files = glob.glob(os.sep.join([files[i], '*']))
            # Get unique filenames
            label_files = list(set(["_".join(file.split('_')[:-1]) for file in label_files]))

            # Filter for channels
            protein_paths = [f"{file}_protein.tif" for file in label_files]
            nucleus_paths = [f"{file}_nucleus.tif" for file in label_files]
            masks_paths = [f"{file}_mask.tif" for file in label_files]
            filtered_df = pd.DataFrame({'protein_paths': protein_paths,
                                        'nucleus_paths': nucleus_paths,
                                        'mask_paths': masks_paths})
            # Assign label
            filtered_df['label'] = labels[i]

            # Accumulate
            accum_df.append(filtered_df)
        df_metadata = pd.concat(accum_df, ignore_index=True)

        return df_metadata

    def create_path_iterators(self) -> List[List[str]]:
        """Return list containing list of absolute paths for each channel."""
        path_gens = []
        for channel in ["protein_paths", "nucleus_paths"]:
            path_gens.append(self.metadata[channel].tolist())
        return path_gens


class CyCLOPSValidation(ValidationProcedure):
    """Yeast Perturbation Dataset Evaluation."""
    def __init__(self, dset):
        self.data_dir = f'/neuhaus/alexlu/datasets/IMAGE_DATASETS/YEAST-PERTURBATION_yolanda-chong/chong_labeled'
        self.metadata = self.load_metadata()
        self.dset = dset

    def load_metadata(self) -> pd.DataFrame:
        """Return metadata dataframe for BBBC021.
        """
        files = glob.glob(os.sep.join([self.data_dir, '*']))
        # Get labels
        labels = [file.split(self.data_dir + os.sep)[-1] for file in files]

        accum_df = []
        for i in range(len(labels)):
            # Get all files for label
            label_files = glob.glob(os.sep.join([files[i], '*']))
            # Get unique filenames
            label_files = list(set(["_".join(file.split('_')[:-1]) for file in label_files]))

            # Filter for channels
            gfp_paths = [f"{file}_rfp.tif" for file in label_files]
            rfp_paths = [f"{file}_gfp.tif" for file in label_files]
            filtered_df = pd.DataFrame({'gfp_paths': gfp_paths,
                                        'rfp_paths': rfp_paths})
            # Assign label
            filtered_df['label'] = labels[i]

            # Accumulate
            accum_df.append(filtered_df)
        df_metadata = pd.concat(accum_df, ignore_index=True)

        return df_metadata

    def create_path_iterators(self) -> List[List[str]]:
        """Return list containing list of absolute paths for each channel."""
        path_gens = []
        for channel in ["gfp_paths", "rfp_paths"]:
            path_gens.append(self.metadata[channel].tolist())
        return path_gens


if __name__ == "__main__" and "D:\\" not in os.getcwd():
    # ==PARAMETERS==
    dset = 'full'

    protocol = BBBC021Protocol(dset)

    for weights in ['cytoimagenet']:
        for concat in [True, False]:
            for norm in [True, False]:
                protocol.evaluate(weights=weights, concat=concat, norm=norm)
        # Get final results
        protocol.get_all_results(weights)
        protocol.plot_all_results()
        print(f"Done with {check_weights_str(weights)}!")

