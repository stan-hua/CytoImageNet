from tensorflow.keras.applications import EfficientNetB0
import tensorflow as tf

from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import faiss

import cv2
import PIL

import os
import glob
import warnings
import datetime
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np
import scipy.stats as stats
from math import ceil

import umap
import seaborn as sns
import matplotlib.pyplot as plt

# Add scripts subdirectory to PATH
import sys


# Plotting Settings
sns.set_style("white")
plt.style.use('seaborn-white')
plt.rc('font', family='serif')

# PATHS
if "D:\\" in os.getcwd():
    annotations_dir = "M:/home/stan/cytoimagenet/annotations/"
    data_dir = 'M:/ferrero/stan_data/'
    evaluation_dir = "M:/home/stan/cytoimagenet/evaluation/"
    embedding_dir = "M:/ferrero/stan_data/evaluation/"
    scripts_dir = "M:/home/stan/cytoimagenet/scripts"
    weights_dir = 'M:/home/stan/cytoimagenet/model/cytoimagenet-weights/'
    plot_dir = "M:/home/stan/cytoimagenet/figures/"
else:
    annotations_dir = "/home/stan/cytoimagenet/annotations/"
    data_dir = '/ferrero/stan_data/'
    evaluation_dir = "/home/stan/cytoimagenet/evaluation/"
    embedding_dir = "/ferrero/stan_data/evaluation/"
    scripts_dir = '/home/stan/cytoimagenet/scripts'
    weights_dir = '/home/stan/cytoimagenet/model/cytoimagenet-weights/'
    plot_dir = "/home/stan/cytoimagenet/figures/"

sys.path.append(f"{scripts_dir}/data_processing")
from preprocessor import normalize as img_normalize

# Only use CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Remove warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def load_model(weights='cytoimagenet',
               weights_filename='efficientnetb0_from_random-epoch_48.h5',
               init='random', overwrite=True, dset_=None):
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
        elif dset_ == 'full_filtered':
            num_classes = 552
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


# faiss kNN Implementation
class FaissKNeighbors:
    """Efficient kNN Implementation using faiss library, following scikit-learn
    conventions.

    Modified from a TowardsDataScience article.
    Link: https://towardsdatascience.com/make-knn-300-times-faster-than-scikit-learns-in-20-lines-5e29d74e76bb

    Cosine similarity code modified from GitHub Issue.
    Link: https://github.com/facebookresearch/faiss/issues/1119#issuecomment-596514782
    """
    def __init__(self, k=5, metric='euclidean'):
        self.metric = metric
        self.index = None
        self.y = None
        self.k = k

    def fit(self, X, y):
        X_copy = X.copy()
        if self.metric == 'euclidean':
            self.index = faiss.IndexFlatL2(X.shape[1])
        else:   # cosine distance
            quantizer = faiss.IndexFlatIP(X.shape[1])
            self.index = faiss.IndexIVFFlat(quantizer, X.shape[1],
                                            int(np.sqrt(X.shape[0])),
                                            faiss.METRIC_INNER_PRODUCT)
            faiss.normalize_L2(X_copy.astype(np.float32))
            self.index.train(X_copy.astype(np.float32))
        self.index.add(X_copy.astype(np.float32))
        self.y = y

    def predict(self, X):
        # Create deep copy
        X_copy = X.copy()

        if self.metric == 'cosine':
            # L2 Normalize
            faiss.normalize_L2(X_copy.astype(np.float32))

        distances, indices = self.index.search(X_copy.astype(np.float32), k=self.k)
        votes = self.y[indices]
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        return predictions

    def kneighbors(self, X, n_neighbors: int):
        if isinstance(X, list):
            X_copy = np.array(X)
        else:
            # Create deep copy
            X_copy = X.copy()

        if self.metric == 'cosine':
            # L2 Normalize
            faiss.normalize_L2(X_copy.astype(np.float32))

        dist, ind = self.index.search(X_copy.astype(np.float32),
                                 k=n_neighbors)

        return dist, ind


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


def check_weights_str(weights, weights_suffix=''):
    if weights is None:     # None -> 'random'
        weights_str = 'random'
    elif weights == 'cytoimagenet':
        weights_str = weights + weights_suffix
    else:
        weights_str = 'imagenet'
    return weights_str


def timer(start, end, print_out=False):
    time_delta = (end - start)
    total_seconds = time_delta.total_seconds()
    minutes = total_seconds/60
    if print_out:
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
        self.read_with = 'PIL'

    def get_image(self, paths: list):
        """Returns channel image/s after image operations specified in
        class attributes.
        """
        imgs = []
        for path in paths:
            # Load images using OpenCV by default. PIL is used otherwise.
            if self.read_with == 'cv2':
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                # Begin reading with PIL if cv2 fails
                if img is None:
                    self.read_with = 'PIL'
                    img = np.array(PIL.Image.open(path))
            else:
                img = np.array(PIL.Image.open(path))

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
        metadata:
            pandas DataFrame containing paths to channel images
        data_dir:
            directory location of dataset
        name:
            name of dataset (convention)
        num_channels:
            number of channels for each image
        path_gens:
            list of <num_channels> path iterators one for each channel
        len_dataset:
            number of <num_channel> unique images in dataset
        k_values:
            tuple of k values to test in kNN
        cytoimagenet_weights_suffix:
            suffix that specifies cytoimagenet weights used
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

        self.cytoimagenet_weights_suffix = ''

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
        accum_activations = model.predict(ds_test, verbose=1,
                                          steps=steps_to_predict, workers=32)

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
        weights_str = check_weights_str(weights, self.cytoimagenet_weights_suffix)

        # Check if directory exists
        if not os.path.exists(f"{embedding_dir}/{weights_str}_embeddings"):
            os.mkdir(f"{embedding_dir}/{weights_str}_embeddings")

        filename = f"{embedding_dir}{weights_str}_embeddings/{self.name}_embeddings"
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
        weights_str = check_weights_str(weights, self.cytoimagenet_weights_suffix)
        filename = f"{embedding_dir}{weights_str}_embeddings/{self.name}_embeddings"
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
            print("Finished Feature Extraction in ", round(total, 2), " minutes!")

            # Log time taken
            if not os.path.exists(evaluation_dir + f"inference_times/{self.name}_inference_times({weights_str}).csv"):
                df_time = pd.DataFrame({'to_grayscale': not concat, 'normalize': norm, 'minutes': total}, index=[0])
                df_time.to_csv(evaluation_dir + f"inference_times/{self.name}_inference_times({weights_str}).csv", index=False)
            else:
                df_time = pd.read_csv(evaluation_dir + f"inference_times/{self.name}_inference_times({weights_str}).csv")
                # Remove past record
                df_time = df_time[(df_time.to_grayscale == concat) | (df_time.normalize != norm)]
                # Add existing time
                df_curr = pd.DataFrame({'to_grayscale': not concat, 'normalize': norm, 'minutes': total}, index=[0])
                df_time = pd.concat([df_time, df_curr])
                df_time.to_csv(evaluation_dir + f"inference_times/{self.name}_inference_times({weights_str}).csv", index=False)

        # Subtract Mean and Divide by Standard Deviation
        scaler = preprocessing.StandardScaler()
        transformed_activations = scaler.fit_transform(activations)

        return pd.DataFrame(transformed_activations)

    # Leave-one-out Classifier
    def knn_classify(self, df_test: np.array, y_test: np.array,
                       k: int, metric='euclidean',
                       weights: str = 'imagenet', concat=True, norm=True,
                       overwrite=False, df_train=None, y_train=None):
        """Perform <k> Nearest Neighbors Classification of test set. Implemented
        using faiss library.

        If <df_train> and <y_train> specified, predict labels in test set based
        on training set. Else, perform a leave-one-out on the test set.

        ==Precondition==:
            - features in <df_test> are centered and standardized by
                mean and std.
        ==Parameters==:
            - df_test is array of (num_samples, num_features)
        """
        weights_str = check_weights_str(weights, self.cytoimagenet_weights_suffix)

        # End early if results already exist
        if not overwrite and os.path.exists(f"{evaluation_dir}/{weights_str}_results/"
                                            f"{self.name}_kNN_results({weights_str}, "
                                            f"{create_save_str(concat, norm)}, k-{k}).csv"):
            return

        # Timing kNN
        start_kNN = datetime.datetime.now()

        # Convert from pandas dataframe to numpy array
        if type(df_train) == pd.DataFrame:
            df_train = df_train.to_numpy()
        if type(df_test) == pd.DataFrame:
            df_test = df_test.to_numpy()

        # Enforce C-Contiguity for arrays
        if type(df_train) == np.ndarray and not df_train.flags['C_CONTIGUOUS']:
            df_train = df_train.copy(order='C')
        if type(df_test) == np.ndarray and not df_test.flags['C_CONTIGUOUS']:
            df_test = df_test.copy(order='C')

        # Encode Labels
        le = preprocessing.LabelEncoder()
        test_labels = le.fit_transform(y_test)

        if y_train is not None:
            train_labels = le.transform(y_train)

        # Prediction/Results Accumulator
        encoded_labels = np.unique(test_labels)
        correct_by_class = np.zeros(len(encoded_labels), dtype=np.uint32)
        total_by_class = np.zeros(len(encoded_labels), dtype=np.uint32)
        correct = 0.0
        total = 0.0

        # Fit on the training set, if specified. Else, fit on the test set.
        knn_model = FaissKNeighbors(k=k, metric=metric)
        if df_train is not None:
            knn_model.fit(df_train, train_labels)
        else:
            knn_model.fit(df_test, test_labels)

        # Time per iteration
        time_per_iter = None

        # Iterate through each sample. Match to existing samples.
        for i in range(len(df_test)):
            # Print progress
            curr_status = round(100 * i / len(df_test), 2)
            if curr_status in range(0, 101, 20):
                progress_str = f"{curr_status}% [{i}/{len(df_test)}]"
                if time_per_iter is not None:
                    mins_remaining = round(time_per_iter * (len(df_test) - i), 0)
                    hours = int(mins_remaining / 60)
                    mins = int(mins_remaining % 60)
                    progress_str += f" Time Remaining: {hours}:{mins}"
                progress_bar = f"[{'='*int(30 * i/len(df_test))}>{'.'*int(30 * (1-(i/len(df_test))))}]"
                print(progress_str)
                print(progress_bar)
            start = datetime.datetime.now()

            # Ignore duplicate point if kNN fitted on test set
            if y_train is None:
                neigh_dist, neigh_ind = knn_model.kneighbors([df_test[i, :]],
                                                                 n_neighbors=k + 1)
                neigh_dist = neigh_dist[0][1:]
                neigh_ind = neigh_ind[0][1:]
                neigh_labels = test_labels[neigh_ind]
            else:
                neigh_dist, neigh_ind = knn_model.kneighbors([df_test[i, :]],
                                                             n_neighbors=k)
                # remove outer list
                neigh_dist = neigh_dist[0]
                neigh_ind = neigh_ind[0]

                neigh_labels = train_labels[neigh_ind]

            predicted = 0
            # if only one value predicted
            if isinstance(neigh_labels, int):
                predicted = neigh_labels
            # elif more than one non-unique label, get mode
            elif len(np.unique(neigh_labels)) < len(neigh_labels):
                predicted = stats.mode(neigh_labels)[0][0]
            # else, take the label of the closest point
            else:
                smallest_ind = np.argmin(neigh_dist)
                predicted = neigh_labels[smallest_ind]

            # Check prediction accuracy
            if predicted == test_labels[i]:
                correct += 1.0
                correct_by_class[test_labels[i]] += 1
            total += 1.0
            total_by_class[test_labels[i]] += 1

            # Time per iteration
            end = datetime.datetime.now()
            time_per_iter = timer(start, end)

        # Save Results
        df_results = pd.DataFrame()
        df_results['labels'] = le.inverse_transform(np.unique(test_labels))
        df_results['correct_by_class'] = correct_by_class
        df_results['total_by_class'] = total_by_class
        df_results['accuracy_by_class'] = correct_by_class / total_by_class
        df_results['total_accuracy'] = correct / total

        # Check if directory exists
        if not os.path.exists(f"{evaluation_dir}/{weights_str}_results"):
            os.mkdir(f"{evaluation_dir}/{weights_str}_results")

        df_results.to_csv(f"{evaluation_dir}/{weights_str}_results/{self.name}_kNN_results({weights_str}, {create_save_str(concat, norm)}, k-{k}).csv", index=False)

        # Log time for kNN Predictions
        end = datetime.datetime.now()
        time_taken = timer(start_kNN, end)
        print(f"Finished {k}-NN prediction in ", round(time_taken, 2), " minutes!")

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
        weights_str = check_weights_str(weights, self.cytoimagenet_weights_suffix)       # converts None -> 'random'

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
        weights_str = check_weights_str(weights, self.cytoimagenet_weights_suffix)       # converts None -> 'random'
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
        all_weights = ['imagenet', f'cytoimagenet{self.cytoimagenet_weights_suffix}', 'random']

        # Get accumulated results
        accum_df = []
        for i in range(len(all_weights)):
            curr_df = pd.read_csv(f"{evaluation_dir}{self.name}_aggregated_results({all_weights[i]}).csv")
            curr_df['ci'] = curr_df.apply(lambda x: 1.96 * np.sqrt((x.total_accuracy * (1 - x.total_accuracy)) / self.len_dataset), axis=1)
            curr_df['preproc_method'] = curr_df.apply(lambda x: create_save_str(not x.to_grayscale, x.normalized), axis=1)
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

                axs[i].set_title(methods[i])
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

        ax.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
        plt.xlabel("")
        plt.ylabel("")
        plt.tick_params(left=False, bottom=False, labelleft=False,
                        labelbottom=False)
        plt.tight_layout()

        # Save Figure
        plot_dir = "/home/stan/cytoimagenet/figures/"
        if save:
            if not os.path.isdir(f"{plot_dir}umap/{self.name}"):
                os.mkdir(f"{plot_dir}umap/{self.name}")
            weights_str = check_weights_str(weights, self.cytoimagenet_weights_suffix)       # converts None -> 'random'
            plt.savefig(f"{plot_dir}umap/{self.name}/{self.name}_features({weights_str}, {create_save_str(concat, norm)}).png",
                        bbox_inches='tight', dpi=400)


class BBBC021Protocol(ValidationProcedure):
    """BBBC021 Evaluation Protocol. 1-nearest neighbors classification of
    mechanism-of-action (MOA). Reports NSC Accuracy.
        NOTE: Aggregates ~2000 feature vectors by 103 treatments to get 103
                feature vectors.
        NOTE: Time for feature extraction is recorded.
    """
    def __init__(self, dset, suffix=''):
        self.data_dir = f"{data_dir}bbbc021"
        self.metadata = self.load_metadata()
        self.name = 'bbbc021'
        self.num_channels = 3
        self.path_gens = self.create_path_iterators()
        self.len_dataset = 103
        self.k_values = (1,)
        self.dset = dset
        self.cytoimagenet_weights_suffix = suffix

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
    def knn_classify(self, df_test, y_test: np.array, compounds: np.array,
                     k: int, metric='cosine', method="nsc",
                     weights: str = 'imagenet', concat=True, norm=True,
                     overwrite=False):
        """Perform <k> Nearest Neighbors Classification leave-one-out of each
        sample, using activations in <df_test>, and labels in <y_test>.
        Implemented using faiss library.

        ==Precondition==:
            - <df_test> activations and <y_test> correspond. As a result, their
                lengths should also be equal.
        ==Parameters==:
            - df_test is array of (num_samples, num_features)
        """
        weights_str = check_weights_str(weights, self.cytoimagenet_weights_suffix)

        # End early if results already exist
        if not overwrite and os.path.exists(f"{evaluation_dir}/{weights_str}_results/"
                                            f"{self.name}_kNN_results({weights_str}, "
                                            f"{create_save_str(concat, norm)}, k-{k}).csv"):
            return
        # Convert pandas dataframe to numpy arrays
        if type(df_test) == pd.DataFrame:
            df_test = df_test.to_numpy()

        # Encode Labels
        le = preprocessing.LabelEncoder()
        test_labels = le.fit_transform(y_test)

        # Prediction/Results Accumulator
        encoded_labels = np.unique(test_labels)
        correct_by_class = np.zeros(len(encoded_labels), dtype=np.uint32)
        total_by_class = np.zeros(len(encoded_labels), dtype=np.uint32)
        correct = 0.0
        total = 0.0

        # Fit KNN Classifier on the entire dataset
        knn_model = KNeighborsClassifier(n_neighbors=k, metric=metric,
                                         algorithm='brute')
        knn_model.fit(df_test, test_labels)

        # Iterate through each sample. Match to existing samples.
        for i in range(len(df_test)):
            # print(f"Classifying MOA: {unproc_labels[i]}, Compound: {compounds[i]}")

            # Get k+1 neighbors of current sample, since 1st point is duplicate
            neigh_dist, neigh_ind = knn_model.kneighbors([df_test[i, :]],
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
                        [df_test[i, :]], n_neighbors=k+1+accum_num_same)

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

            neigh_labels = test_labels[neigh_ind]

            # If there is at least one non-unique value, take the mode, otherwise take the closest
            predicted = 0
            if not isinstance(neigh_labels, np.ndarray) and not isinstance(neigh_labels, list):    # if only one value
                predicted = neigh_labels
            elif len(np.unique(neigh_labels)) < len(neigh_labels):  # if more than one non-unique label, get mode
                predicted = stats.mode(neigh_labels)[0][0]
            else:
                smallest_ind = np.argmin(neigh_dist)
                predicted = neigh_labels[smallest_ind]

            # Check predictions
            if predicted == test_labels[i]:
                correct += 1.0
                correct_by_class[test_labels[i]] += 1
            total += 1.0
            total_by_class[test_labels[i]] += 1

        # Save Results
        df_results = pd.DataFrame()
        df_results['labels'] = le.inverse_transform(np.unique(test_labels))
        df_results['correct_by_class'] = correct_by_class
        df_results['total_by_class'] = total_by_class
        df_results['accuracy_by_class'] = correct_by_class / total_by_class
        df_results['total_accuracy'] = correct / total

        if not os.path.exists(f"{evaluation_dir}/{weights_str}_results"):
            os.mkdir(f"{evaluation_dir}/{weights_str}_results")

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
                              norm=norm, weights=weights, overwrite=True)

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

        dset: convention for subset of CytoImageNet used to train model. Not to
                be confused with the method argument 'dset' which refers to
                either train or test set for the COOS-7 dataset
    """
    def __init__(self, dset, test, suffix=''):
        self.data_dir = f'/neuhaus/alexlu/datasets/IMAGE_DATASETS/COOS-MOUSE_david-andrews/CURATED-COOS7/{test}'
        self.test_metadata = self.load_metadata(coos_dset='test')
        self.train_metadata = self.load_metadata(coos_dset='train')
        self.test_set = test
        self.name = f'coos7_{test}'
        self.num_channels = 2
        self.path_gens = self.create_path_iterators()
        self.len_dataset = len(self.path_gens[0])
        self.k_values = (11,)
        self.dset = dset

        self.cytoimagenet_weights_suffix = suffix

    def load_metadata(self, coos_dset='test') -> pd.DataFrame:
        """Return metadata dataframe for COOS-7 based on directory structure..
        """
        # Get labels
        if coos_dset == 'test':
            files = glob.glob(os.sep.join([self.data_dir, '*']))
            labels = [file.split(self.data_dir + os.sep)[-1] for file in files]
        else:
            train_dir = f'/neuhaus/alexlu/datasets/IMAGE_DATASETS/COOS-MOUSE_david-andrews/CURATED-COOS7/train'
            files = glob.glob(os.sep.join([train_dir, '*']))
            labels = [file.split(train_dir + os.sep)[-1] for file in files]

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

    def create_path_iterators(self, coos_dset='test') -> List[List[str]]:
        """Return list containing list of absolute paths for each channel."""
        if coos_dset == 'test':
            metadata = self.test_metadata
        else:
            metadata = self.train_metadata
        path_gens = []
        for channel in ["protein_paths", "nucleus_paths"]:
            path_gens.append(metadata[channel].tolist())
        return path_gens

    def evaluate(self, weights, concat, norm, overwrite=False):
        """Main function to carry out evaluation protocol.
        If overwrite, ignore existing embeddings and extract.
        """
        df_train, df_test = self.load_activations(concat, norm, weights,
                                                  overwrite=overwrite)
        y_train = self.train_metadata['label'].to_numpy()
        y_test = self.test_metadata['label'].to_numpy()

        # kNN Classifier
        for k in self.k_values:
            self.knn_classify(df_test, y_test, k, concat=concat,
                              norm=norm, weights=weights,
                              df_train=df_train, y_train=y_train,
                              overwrite=True)

        # Create UMAP visualization of features
        self.umap_visualize(df_test, labels=y_test, weights=weights,
                            concat=concat, norm=norm, save=True)

    # Embeddings
    def load_activations(self, concat=True, norm=False, weights="imagenet",
                         overwrite=False) -> Tuple[np.array, np.array]:
        """Return tuple of 2 activation matrices (samples, features),
        corresponding to training and test set embeddings, respectively.

        Activations are centered and standardized with respect to the training
        set.

        If <overwrite>, extract embeddings even if embeddings already exist.
        """
        # Filename of embeddings
        weights_str = check_weights_str(weights, self.cytoimagenet_weights_suffix)

        # Get activations for train set and test set
        for dset in ['train', 'test']:
            if dset == 'train':
                name = 'coos7_train'
            else:
                name = self.name

            filename = f"{embedding_dir}{weights_str}_embeddings/{name}_embeddings"
            suffix = f" ({weights_str}, {create_save_str(concat, norm)})"
            # Load embeddings if present
            if os.path.exists(f"{filename}{suffix}.h5") and not overwrite:
                activations = pd.read_hdf(f"{filename}{suffix}.h5", 'embed')
            else:
                # If not, extract embeddings
                print(f"Beginning extraction of {name.upper()} w/{suffix}...")
                start = datetime.datetime.now()
                activations = self.extract_embeddings(concat, norm, weights, overwrite=overwrite, coos_dset=dset)
                end = datetime.datetime.now()
                total = timer(start, end)
                print("Finished Feature Extraction in ", round(total, 2), " minutes!")

                # Log time taken
                if not os.path.exists(evaluation_dir + f"inference_times/{name}_inference_times({weights_str}).csv"):
                    df_time = pd.DataFrame({'to_grayscale': not concat, 'normalize': norm, 'minutes': total}, index=[0])
                    df_time.to_csv(evaluation_dir + f"inference_times/{self.name}_inference_times({weights_str}).csv", index=False)
                else:
                    df_time = pd.read_csv(evaluation_dir + f"inference_times/{name}_inference_times({weights_str}).csv")
                    # Remove past record
                    df_time = df_time[(df_time.to_grayscale == concat) | (df_time.normalize != norm)]
                    # Add existing time
                    df_curr = pd.DataFrame({'to_grayscale': not concat, 'normalize': norm, 'minutes': total}, index=[0])
                    df_time = pd.concat([df_time, df_curr])
                    df_time.to_csv(evaluation_dir + f"inference_times/{name}_inference_times({weights_str}).csv", index=False)

            # Reassign variable
            if dset == 'train':
                df_train = activations
            else:
                df_test = activations

        # Subtract Mean and Divide by Standard Deviation
        scaler = preprocessing.StandardScaler()
        train_scaled = scaler.fit_transform(df_train)
        test_scaled = scaler.transform(df_test)

        return (train_scaled, test_scaled)

    def extract_embeddings(self, concat=True, norm=False, weights="imagenet",
                           overwrite=False, coos_dset='test') -> pd.DataFrame:
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
        test_generator = ImageGenerator(self.create_path_iterators(
            coos_dset=coos_dset), concat, norm)
        ds_test = tf.data.Dataset.from_generator(lambda: test_generator, output_types=(tf.float32))

        # Prefetch data in batches
        batch_size = 1
        # ds_test = ds_test.batch(batch_size)
        ds_test = ds_test.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        steps_to_predict = ceil(len(test_generator) / batch_size)
        # Extract embeddings
        accum_activations = model.predict(ds_test, verbose=1,
                                          steps=steps_to_predict, workers=32)

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
        weights_str = check_weights_str(weights, self.cytoimagenet_weights_suffix)

        # Check if directory exists
        if not os.path.exists(f"{embedding_dir}/{weights_str}_embeddings"):
            os.mkdir(f"{embedding_dir}/{weights_str}_embeddings")

        if coos_dset == 'test':
            name = self.name
        else:
            name = 'coos7_train'

        filename = f"{embedding_dir}{weights_str}_embeddings/{name}_embeddings"
        filename += f" ({weights_str}, {create_save_str(concat, norm)})"

        activations.to_hdf(f"{filename}.h5", key='embed', index=False)
        return activations


class CyCLOPSValidation(ValidationProcedure):
    """Yeast Perturbation Dataset Evaluation."""
    def __init__(self, dset, suffix=''):
        self.data_dir = f'/neuhaus/alexlu/datasets/IMAGE_DATASETS/YEAST-PERTURBATION_yolanda-chong/chong_labeled'
        self.metadata = self.load_metadata()
        self.name = 'cyclops'
        self.num_channels = 2
        self.path_gens = self.create_path_iterators()
        self.len_dataset = len(self.path_gens[0])
        self.k_values = (11, )
        self.dset = dset

        self.cytoimagenet_weights_suffix = suffix

    def load_metadata(self) -> pd.DataFrame:
        """Return metadata dataframe for CyCLOPS based on directory structure.
        """
        files = glob.glob(os.sep.join([self.data_dir, '*']))
        # Get labels
        labels = [file.split(self.data_dir + os.sep)[-1] for file in files]
        labels.remove('DEAD')
        labels.remove('GHOST')

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
            # Assign label. Merge spindle and spindlepole
            if labels[i] == 'SPINDLEPOLE':
                filtered_df['label'] = 'SPINDLE'
            else:
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

    def evaluate(self, weights, concat, norm, overwrite=True):
        """Main function to carry out evaluation protocol.
        If overwrite, ignore existing embeddings and extract.
        """
        df_activations = self.load_activations(concat, norm, weights, overwrite=overwrite)
        labels = self.metadata['label'].to_numpy()

        if len(df_activations) != len(labels):
            print(f"{len(df_activations)} != {len(labels)}")
            df_activations = self.load_activations(concat, norm, weights, overwrite=True)


        # kNN Classifier
        for k in self.k_values:
            self.knn_classify(df_activations, labels, k, concat=concat,
                              norm=norm, weights=weights, overwrite=True)

        # Create UMAP visualization of features
        self.umap_visualize(df_activations, labels=labels, weights=weights,
                            concat=concat, norm=norm, save=True)


def main_coos(cyto_suffix='', dset='full'):
    for i in range(1, 5):
        start = datetime.datetime.now()
        protocol = COOS7Validation(dset, test=f"test{i}")
        protocol.cytoimagenet_weights_suffix = cyto_suffix
        print("=" * 30)
        print(f"COOS-7 Test Set {i} Processing...")
        print("=" * 30)
        for weights in ['cytoimagenet']:
            for concat in [True, False]:
                for norm in [True, False]:
                    suffix = f" ({check_weights_str(weights)}, {create_save_str(concat, norm)})"
                    print(f"Processing kNN predictions for {protocol.name.upper()} w/{suffix}...")
                    protocol.evaluate(weights=weights, concat=concat, norm=norm)
            # Get final results
            protocol.get_all_results(weights)
            print(f"Done with {check_weights_str(weights)}!")
        end = datetime.datetime.now()
        total = timer(start, end)
        print(f"COOS-7 Test Set {i} finished in {round(total, 2)} minutes..")

    # Create plot comparing ImageNet, CytoImageNet and Random features
    protocol.plot_all_results()


def main_cyclops(cyto_suffix='', dset='full'):
    start = datetime.datetime.now()
    protocol = CyCLOPSValidation(dset)
    protocol.cytoimagenet_weights_suffix = cyto_suffix
    print("=" * 30)
    print(f"CyCLOPS Processing...")
    print("=" * 30)
    for weights in ['cytoimagenet']:
        for concat in [True, False]:
            for norm in [True, False]:
                suffix = f" ({check_weights_str(weights)}, {create_save_str(concat, norm)})"
                print(f"Processing kNN predictions for {protocol.name.upper()} w/{suffix}...")
                protocol.evaluate(weights=weights, concat=concat, norm=norm,
                                  overwrite=False)
        # Get final results
        protocol.get_all_results(weights)
        print(f"Done with {check_weights_str(weights)}!")
    end = datetime.datetime.now()
    total = timer(start, end)
    print(f"CyCLOPS finished in {round(total, 2)} minutes..")

    # Create plot comparing ImageNet, CytoImageNet and Random features
    protocol.plot_all_results()


def main_bbbc021(cyto_suffix='', dset='full'):
    start = datetime.datetime.now()
    protocol = BBBC021Protocol(dset)
    protocol.cytoimagenet_weights_suffix = cyto_suffix
    print("=" * 30)
    print(f"BBBC021 Processing...")
    print("=" * 30)
    for weights in ['cytoimagenet']:
        for concat in [True, False]:
            for norm in [True, False]:
                if concat and norm:
                    continue
                suffix = f" ({check_weights_str(weights)}, {create_save_str(concat, norm)})"
                print(f"Processing kNN predictions for {protocol.name.upper()} w/{suffix}...")
                protocol.evaluate(weights=weights, concat=concat, norm=norm)
        # Get final results
        protocol.get_all_results(weights)
        print(f"Done with {check_weights_str(weights)}!")
    end = datetime.datetime.now()
    total = timer(start, end)
    print(f"BBBC021 finished in {round(total, 2)} minutes..")

    # Create plot comparing ImageNet, CytoImageNet and Random features
    protocol.plot_all_results()


def redo_get_results():
    files = glob.glob(f'{evaluation_dir}/bbbc021_aggregated*.csv')

    for file in files:
        df_results = pd.read_csv(file)
        df_results['ci'] = 1.96 * np.sqrt((df_results['total_accuracy'] * (1 - df_results['total_accuracy'])) / 103)
        df_results.to_csv(file, index=False)


def compile_results(val_set='bbbc021'):
    files = glob.glob(f"{evaluation_dir}{val_set}_aggregated_results(*).csv")

    if val_set == 'bbbc021':
        files = [i for i in files if 'toy' not in i]    # remove toy examples
        files = [i for i in files if '(cytoimagenet).csv' not in i]

    # Change naming convention
    map_name = {'cytoimagenetfull-16_epochs': 'CytoImageNet-894 [16 epochs]',
                'cytoimagenetfull-100_epochs': 'CytoImageNet-894 [100 epochs]',
                'cytoimagenetfull(aug)-24_epochs': 'CytoImageNet-894 with aug [24 epochs]',
                'cytoimagenetfull(aug)-48_epochs': 'CytoImageNet-894 with aug [48 epochs]',
                'cytoimagenetfull_filtered-43_epochs': 'CytoImageNet-552 with aug [43 epochs]',
                'cytoimagenetfull_filtered-80_epochs': 'CytoImageNet-552 with aug [80 epochs]',
                'cytoimagenetfull_filtered(no_aug)-20_epochs': 'CytoImageNet-552 w/o aug [20 epochs]',
                'random': "Random", "imagenet": "ImageNet",
                }

    curr_df = pd.read_csv(files[0])
    accum_df = pd.DataFrame()
    accum_df['Accuracy by preprocessing method'] = curr_df.apply(
        lambda x: create_save_str(not x.to_grayscale, x.normalized).replace(', ', ' and '), axis=1)
    accuracy_ci = curr_df.apply(lambda x: f"{round(x.total_accuracy * 100, 2)} +/- {round(x.ci * 100, 2)}%", axis=1).tolist()
    accum_df = accum_df.assign(**{map_name[curr_df.weights.iloc[0]]: accuracy_ci})

    for file in files[1:]:
        curr_df = pd.read_csv(file)
        accuracy_ci = curr_df.apply(lambda x: f"{round(x.total_accuracy * 100, 2)} +/- {round(x.ci * 100, 2)}%", axis=1).tolist()
        accum_df = accum_df.assign(**{map_name[curr_df.weights.iloc[0]]: accuracy_ci})

    # Reorder columns
    preferred = ['Accuracy by preprocessing method', 'Random', 'ImageNet',
                 'CytoImageNet-894 with aug [48 epochs]',
                 # 'CytoImageNet-894 with aug [24 epochs]',
                 # 'CytoImageNet-894 [16 epochs]', 'CytoImageNet-894 [100 epochs]',
                 # 'CytoImageNet-552 with aug [43 epochs]', 'CytoImageNet-552 with aug [80 epochs]',
                 # 'CytoImageNet-552 w/o aug [20 epochs]'
                 ]
    preferred = [i for i in preferred if i in accum_df.columns]
    accum_df= accum_df[preferred]

    accum_df.to_csv(f"{evaluation_dir}/{val_set}_final_results.csv", index=False)


if __name__ == "__main__" and "D:\\" not in os.getcwd():
    dset = 'full'

    # Weights for models trained on 894 class dataset
    weights_full_overfit = 'efficientnetb0_from_random(lr_0001_bs_64_epochs_100).h5'
    weights_full_underfit = 'efficientnetb0_from_random-epoch_16.h5'
    # weights_full_aug = 'efficientnetb0_from_random-epoch_24.h5'
    weights_full_aug = 'efficientnetb0_from_random-epoch_48.h5'

    # Weights for models trained on 552 class dataset
    weights_full_filtered_underfit = 'efficientnetb0_from_random-epoch_43.h5'
    weights_full_filtered_overfit = 'efficientnetb0_from_random-epoch_80.h5'
    weights_full_filtered_noaug = 'efficientnetb0_from_random-epoch_20.h5'

    main_coos('full(aug)-48_epochs', dset)
    main_cyclops('full(aug)-48_epochs', dset)
    main_bbbc021('full(aug)-48_epochs', dset)

    for val_set in ['bbbc021', 'cyclops', 'coos7_test1', 'coos7_test2', 'coos7_test3', 'coos7_test4']:
        compile_results(val_set)
