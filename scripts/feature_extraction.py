import multiprocessing
import multiprocessing
import os
import random
from itertools import combinations
from math import ceil

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from numpy import dot
from numpy.linalg import norm
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from data_curation.analyze_metadata import get_df_counts
from model_evaluation import load_model

# Set CPU only
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Global Variables
if "D:\\" in os.getcwd():
    annotations_dir = "M:/home/stan/cytoimagenet/annotations/"
    data_dir = 'M:/ferrero/stan_data/'
    evaluation_dir = "M:/home/stan/cytoimagenet/evaluation/"
    model_dir = "M:/home/stan/cytoimagenet/model/"
    scripts_dir = "M:/home/stan/cytoimagenet/scripts"
    weights_dir = 'M:/home/stan/cytoimagenet/model/cytoimagenet-weights/'
    plot_dir = "M:/home/stan/cytoimagenet/figures/"
else:
    annotations_dir = "/home/stan/cytoimagenet/annotations/"
    data_dir = '/ferrero/stan_data/'
    evaluation_dir = "/home/stan/cytoimagenet/evaluation/"
    model_dir = "/home/stan/cytoimagenet/model/"
    scripts_dir = '/home/stan/cytoimagenet/scripts'
    weights_dir = '/home/stan/cytoimagenet/model/cytoimagenet-weights/'
    plot_dir = "/home/stan/cytoimagenet/figures/"

# Plot Settings
sns.set_style("dark")
plt.style.use('dark_background')

# Labels
toy_50 = ['104-001', 'actin', 'actin inhibitor', 'active sars-cov-2',
          'activin-a', 'alpha-adrenergic receptor agonist',
          'angiopoietin-1', 'atp-sensitive potassium channel blocker',
          'balf', 'bmp-2', 'calr-kdel', 'cell body', 'cell membrane',
          'cell junctions', 'centrosome', 'chr2 targeted', 'clarithromycin',
          'coco', 'cyclooxygenase inhibitor', 'dmso',
          'dna synthesis inhibitor', 'dydrogesterone',
          'erbb family inhibitor', 'estrogen receptor antagonist',
          'ezh2 targeted', 'fluorometholone', 'glycosylase inhibitor',
          'heat shock protein inhibitor', 'human', 'ifn-gamma', 'kinase',
          'microtubules', 'mitochondria', 'mitotic spindle',
          'nachr antagonist', 'nucleus', 'pcna',
          'rho associated kinase inhibitor', 'rna',
          'rna synthesis inhibitor', 's35651', 'siglec-1', 'tamoxifen',
          'tankyrase inhibitor', 'topoisomerase inhibitor', 'trophozoite',
          'u2os', 'vamp5 targeted',
          'voltage-gated potassium channel blocker', 'yeast']

toy_20 = ['actin', 'cell body', 'cell membrane', 'dmso', 'human', 'kinase',
          'microtubules', 'mitochondria', 'nucleus',
          'rho associated kinase inhibitor', 'rna',
          'rna synthesis inhibitor', 's35651', 'tamoxifen',
          'tankyrase inhibitor', 'trophozoite', 'u2os',
          'uv inactivated sars-cov-2', 'vamp5 targeted', 'yeast']


def timer(start, end):
    """Returns the minutes it took to run a piece of code given the start time
    and end time."""
    time_delta = (end - start)
    total_seconds = time_delta.total_seconds()
    minutes = total_seconds / 60
    print("Finished in ", round(minutes, 2), " minutes!")
    return minutes


# DATA GENERATOR for NON-CytoImageNet Images
def test_datagen(label: str, label_dir, data_subset="train"):
    """Return dataframe of paths compatible with tensorflow keras'
    ImageDataGenerator.flow_from_dataframe()

    If <data_subset> is 'val', return path generator for classes not in
    CytoImageNet. Note that these classes were rejected, but images can come
    from datasets present in CytoImageNet.

    NOTE: This function is used to load images that are not in CytoImageNet.
    """
    if data_subset != 'val':
        if label == 'toy_20':  # use labels from toy_20, not upsampled
            accum_df = []
            for l in toy_20:
                curr_df = pd.read_csv(
                    f"{annotations_dir}classes/{label_dir}/{l}.csv")
                # Check if exists
                exists = curr_df.apply(
                    lambda x: os.path.exists(x.path + "/" + x.filename), axis=1)
                curr_df = curr_df[exists]
                print(len(curr_df), ' exist for ', l)
                # Sample 100 from each class
                accum_df.append(curr_df.sample(n=100))
            df = pd.concat(accum_df, ignore_index=True)
        else:
            df = pd.read_csv(
                f"{annotations_dir}classes/{label_dir}/{label}.csv")
    else:
        df = pd.read_csv("/ferrero/stan_data/unseen_classes/metadata.csv")
    df['full_name'] = df.apply(lambda x: f"{x.path}/{x.filename}", axis=1)
    return df


def load_img(x):
    """Load image using Open-CV. Resize to (224, 224)"""
    img = cv2.imread(x)
    if img is None:
        return
    img_resized = cv2.resize(img, (224, 224), cv2.INTER_LINEAR)
    print("Shape: ", img_resized.shape)
    return img_resized


def extract_all_embeds(model, label: str,
                       directory: str = "imagenet-activations/base",
                       label_dir: str = "", method='gen', data_subset="train",
                       weights='imagenet', dset='full'):
    """Load <model>. Extract activations for all images under label.
    """
    if method == 'gen':
        # Create Image Generator for <label>
        test_gen = ImageDataGenerator().flow_from_dataframe(
            dataframe=test_datagen(label, label_dir, data_subset),
            directory=None,
            x_col='full_name',
            batch_size=1,
            target_size=(224, 224),
            color_mode='rgb',
            shuffle=False,
            class_mode=None,
            interpolation='bilinear',
            seed=1,
            validate_filenames=False
        )
        activations = model.predict(test_gen)
    else:  # iterative method
        # Get metadata for images to extract features from
        df_test = test_datagen(label, label_dir, data_subset)

        # Load images using 30 cores
        pool = multiprocessing.Pool(30)
        try:
            imgs = pool.map(load_img, df_test.full_name.tolist())
            pool.close()
            pool.join()
        except:
            pool.close()

        # Remove Nones for images that were not found/improperly read
        idx_none = []
        final_imgs = []
        for i in range(len(imgs)):
            if imgs[i] is None:
                idx_none.append(i)
            else:
                final_imgs.append(imgs[i])
        final_imgs = np.array(final_imgs)

        # Extract embeddings
        activations = model.predict(final_imgs)

    # Save activations
    if not os.path.exists(f"{model_dir}{directory}"):
        os.mkdir(f"{model_dir}{directory}")
    if data_subset != 'val':
        pd.DataFrame(activations).to_csv(
            f"{model_dir}{directory}/{label}_activations.csv", index=False)
    else:
        pd.DataFrame(activations).to_csv(
            f"{model_dir}{directory}/unseen_classes_embeddings ({weights}, {dset}).csv",
            index=False)
    return activations


# DATA GENERATOR for CytoImageNet Images
def load_dataset(dset: str = 'full'):
    """Return tuple of (training data iterator, labels). Sampling 100 from all
    classes in CytoImageNet

    With following fixed parameters:
        - target size: (224, 224)
        - color mode: RGB
        - shuffle: True
        - seed: 7779836983
        - interpolation: bilinear
    """
    global toy_50, toy_20
    # Sample 100 images from each label
    df_metadata = pd.read_csv('/ferrero/cytoimagenet/metadata.csv')
    df = df_metadata.groupby(by=['label']).sample(n=100, random_state=728565)
    df['full_path'] = df.path + "/" + df.filename

    # Filter for labels
    if dset == 'toy_50':
        df = df[df.label.isin(toy_50)]
    elif dset == 'toy_20':
        df = df[df.label.isin(toy_20)]

    # Create generator
    datagen = ImageDataGenerator()
    train_generator = datagen.flow_from_dataframe(
        dataframe=df,
        directory=None,
        x_col='full_path',
        batch_size=64,
        target_size=(224, 224),
        interpolation="bilinear",
        class_mode=None,
        color_mode="rgb",
        shuffle=False,
        seed=728565
    )
    return train_generator, df.label.tolist()


def extract_embeds_from_sample(dset, weights='cytoimagenet'):
    """Sample 100 images from each class. Extract embeddings for all samples and
    save in {model_dir}/imagenet-activations/full_dset_embeddings.csv"""
    global model_dir
    if weights == 'cytoimagenet':
        if dset == 'toy_20':
            weights_str = 'efficientnetb0_from_random(lr_0001_bs_64).h5'
        else:
            weights_str = 'efficientnetb0_from_random-epoch_60.h5'
    else:
        weights_str = weights
    # Load EfficientNetB0
    model = load_model(weights=weights, weights_filename=weights_str,
                       dset_=dset)

    # Load sampled images
    datagen, labels = load_dataset(dset)
    ds_test = tf.data.Dataset.from_generator(lambda: datagen,
                                             output_types=(tf.float32))
    ds_test = ds_test.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # Number of Steps till done Predicting
    steps_to_predict = ceil(datagen.n / datagen.batch_size)
    print("Beginning feature extraction...")
    embeds = model.predict(ds_test, steps=steps_to_predict, workers=32,
                           verbose=1)
    print("Done extracting!")
    sampled_embeds = pd.DataFrame(embeds)

    assert len(sampled_embeds) == len(labels)
    print("Embeds and labels are of same length!")
    print('Saving features...')
    sampled_embeds.to_hdf(
        f"{model_dir}/{weights}-activations/{dset}_dset_embeddings.h5",
        key='embed', index=False)
    pd.Series(labels).to_hdf(
        f"{model_dir}/{weights}-activations/{dset}_dset_embeddings.h5",
        key='label', index=False)
    print("Done saving features!")


# SIMILARITY METRICS
class DiversityMetric:
    def __init__(self, dset, weights):
        self.dset = dset
        self.weights = weights

    def cosine_similarity(self, img1_img2: tuple) -> float:
        """Return cosine similarity between image features of <img1_img2>[0] and
        <img1_img2>[1].
        """
        img1, img2 = img1_img2
        return dot(img1, img2) / (norm(img1) * norm(img2))

    def intra_similarity(self, label_features: np.array, n=10000) -> list:
        """Calculate intra cosine similarities between images images (features) from one
        label. Estimate by calculating cosine similarity for at most <n> pairs.

        Return list of pairwise cosine similarities.

        ==Parameters==:
            label_features: np.array where each row is a feature vector for one
                            image
        """
        pairs = np.array(list(combinations(label_features, 2)))
        if len(pairs) > n:
            pairs = pairs[np.random.choice(len(pairs), n, replace=False)]

        cos_sims = []
        for pair in pairs:
            cos_sims.append(self.cosine_similarity(pair))
        return cos_sims

    def inter_similarity(self, label_features: np.array,
                         other_features: np.array,
                         n=10000) -> list:
        """Calculate inter cosine similarities between images (features) from a
        label and images from all other labels. Estimate by calculating cosine
        similarity for at most <n> pairs.

        Return list of pairwise cosine similarities.

        ==Parameters==:
            label_features: np.array where each row is a feature vector for an image
                            in label
            other_features: np.array where each row is a feature vector for
                            remaining labels
        """
        cos_sims = []
        for i in range(n):
            cos_sims.append(self.cosine_similarity(
                (random.choice(label_features), random.choice(other_features))))
        return cos_sims

    def calculate_diversity(self, label: str, plot=False):
        """Calculate inter-class diversity and intra-class diversity using pairwise
        cosine similarity. Return mean and SD for inter and intra-class diversity.

        ==Parameters==:
            label: CytoImageNet class to calculate diversity for
        """
        global model_dir

        # Verify that embeddings for samples of the full dataset exists
        if not os.path.exists(
                f"{model_dir}/{self.weights}-activations/{self.dset}_dset_embeddings.h5"):
            extract_embeds_from_sample(self.dset, weights=self.weights)
        if self.weights == 'cytoimagenet':
            suffix = '(16_epochs)'
        else:
            suffix = ''
        df_embeds = pd.read_hdf(
            f"{model_dir}/{self.weights}-activations/{self.dset}_dset_embeddings{suffix}.h5",
            'embed')
        labels = pd.read_hdf(
            f"{model_dir}/{self.weights}-activations/{self.dset}_dset_embeddings{suffix}.h5",
            'label')
        df_embeds['label'] = labels

        # ==INTRA DIVERSITY==:
        # Calculate mean pairwise cosine similarity among 100 samples of the label
        label_vecs = df_embeds[df_embeds.label == label].drop(
            columns=['label']).to_numpy()
        other_label_vecs = df_embeds[df_embeds.label != label].drop(
            columns=['label']).to_numpy()
        intra_similarities = self.intra_similarity(label_vecs)

        # ==INTER DIVERSITY==:
        # Calculate mean pairwise cosine similarity between representative feature vector of label and other labels
        inter_similarities = self.inter_similarity(label_vecs, other_label_vecs)

        if np.random.choice(range(10), 1)[0] == 1 or plot:
            fig, (ax1, ax2) = plt.subplots(2, 1)
            sns.kdeplot(x=intra_similarities,
                        hue=['Intra Diversity'] * len(intra_similarities),
                        ax=ax1)
            sns.kdeplot(x=inter_similarities,
                        hue=['Inter Diversity'] * len(inter_similarities),
                        ax=ax2)
            ax1.set_xlabel("")
            ax1.set_xlim(-0.05, 1)
            ax2.set_xlabel("")
            ax2.set_xlim(-0.05, 1)
            plt.savefig(
                f"{plot_dir}classes/similarity/{label}_diversity({self.dset}).png")
            plt.close(fig)

        return (np.mean(intra_similarities), np.std(intra_similarities)), (
        np.mean(inter_similarities), np.std(inter_similarities))

    def get_div(self, label):
        """Return pd.DataFrame containing diversity metrics for <label>."""
        (intra_mean, intra_sd), (
        inter_mean, inter_sd) = self.calculate_diversity(label)
        curr_div = pd.DataFrame(
            {"label": label, "intra_cos_distance_MEAN": 1 - intra_mean,
             "intra_SD": intra_sd, "inter_cos_distance_MEAN": 1 - inter_mean,
             "inter_SD": inter_sd}, index=[0])

        return curr_div

    def get_all_diversity(self, kind=''):
        """Save dataframe of summary stats for pairwise cosine similarities for
        all 894 CytoImageNet labels.
            - mean and std of INTRA-label pairwise cosine similarities
            - mean and std of INTER-label pairwise cosine similarities
        """
        global toy_50, toy_20
        df_metadata = pd.read_csv("/ferrero/cytoimagenet/metadata.csv")

        if self.dset == 'toy_50':
            df_metadata = df_metadata[df_metadata.label.isin(toy_50)]
        elif self.dset == 'toy_20':
            df_metadata = df_metadata[df_metadata.label.isin(toy_20)]
        # try:
        #     pool = multiprocessing.Pool(16)
        #     accum_div = pool.map(get_div, df_metadata.label.unique().tolist())
        #     pool.close()
        #     pool.join()
        # except Exception as e:
        #     pool.close()
        #     raise Exception(e.message)
        accum_div = []
        for label in df_metadata.label.unique().tolist():
            try:
                accum_div.append(self.get_div(label))
            except:
                print(label, ' errored!')

        df_diversity = pd.concat(accum_div, ignore_index=True)
        df_diversity.to_csv(
            f'{model_dir}similarity/{kind}{self.dset}_diversity({self.weights}).csv',
            index=False)

        return df_diversity

    def plot_all_diversity(self, kind=''):
        """Plot intra cosine distance vs. inter cosine distance for all classes
        in CytoImageNet"""
        if os.path.exists(
                f'{model_dir}similarity/{kind}{self.dset}_diversity({self.weights}).csv'):
            df_diversity = pd.read_csv(
                f'{model_dir}similarity/{kind}{self.dset}_diversity({self.weights}).csv')
        else:
            df_diversity = self.get_all_diversity()

        # Get Category Label
        df_counts = get_df_counts()
        mapping_label = dict(zip(df_counts.label, df_counts.category))
        df_diversity['category'] = df_diversity.label.map(
            lambda x: mapping_label[x])

        fig, ax = plt.subplots()
        sns.scatterplot(x='intra_cos_distance_MEAN',
                        y='inter_cos_distance_MEAN',
                        hue='category', data=df_diversity, legend="full",
                        alpha=1, palette="tab20", s=2, linewidth=0, ax=ax)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
        plt.xlabel('Mean Intra Cosine Distance')
        plt.ylabel('Mean Inter Cosine Distance')
        plt.title(f'{self.weights.capitalize()} Features')
        plt.tight_layout()
        plt.savefig(
            f"{plot_dir}classes/{self.dset}_similarity({self.weights}).png")


def main():
    # Choose model
    weights = 'cytoimagenet'  # 'imagenet' or None
    cyto_weights_file = "efficientnetb0_from_random(lr_0001_bs_64).h5"
    dset = 'toy_20'
    data_subset = 'train'  # use unseen classes?
    num_classes = 20
    method_data_loading = 'iter'  # 'gen' vs 'iter'

    label_dir = ''

    # Directory to save activations
    if weights is None:
        activation_loc = "random_model-activations/"
    elif weights == "cytoimagenet":
        activation_loc = "cytoimagenet-activations/"
    else:
        activation_loc = "imagenet-activations/"

    # Construct Model
    if weights == "cytoimagenet":
        cyto_weights_dir = "/home/stan/cytoimagenet/model/cytoimagenet-weights/random_init/"
        nohead_name = f"{cyto_weights_dir}/{dset}/{cyto_weights_file.replace('.h5', '-notop.h5')}"
        # If weights with no head does not exist
        if True or not os.path.exists(nohead_name):
            # Add efficientnet architecture
            efficientnet_model = EfficientNetB0(weights=None,
                                                include_top=False,
                                                input_shape=(224, 224, 3),
                                                pooling="avg")
            x = Dense(num_classes, activation="softmax")(
                efficientnet_model.output)
            old_model = tf.keras.Model(efficientnet_model.input, x)
            old_model.load_weights(
                f"{cyto_weights_dir}/{dset}/{cyto_weights_file}")
            # Remove prediction layer
            new_model = tf.keras.Model(old_model.input,
                                       old_model.layers[-2].output)
            new_model.save_weights(nohead_name)
        model = EfficientNetB0(weights=nohead_name,
                               include_top=False,
                               input_shape=(None, None, 3),
                               pooling="avg")
    else:
        model = EfficientNetB0(weights=weights,
                               include_top=False,
                               input_shape=(None, None, 3),
                               pooling="avg")
        model.trainable = False

    extract_all_embeds(model, 'toy_20', activation_loc,
                       method=method_data_loading,
                       data_subset=data_subset, label_dir=label_dir,
                       weights=weights, dset=dset)


if __name__ == "__main__" and "D:\\" not in os.getcwd():
    df = pd.read_csv("/ferrero/cytoimagenet/metadata.csv")
    try:
        df.drop(columns=["duplicate"], inplace=True)
    except:
        pass
    df.to_csv("/ferrero/cytoimagenet/metadata.csv", index=False)
    df.to_feather("/ferrero/cytoimagenet/metadata.ftr")

    # dset = 'full'
    # for weights in ['cytoimagenet']:
    #     div_metrics = DiversityMetric(dset=dset, weights=weights)
    #     div_metrics.get_all_diversity()
    #     div_metrics.plot_all_diversity()
else:
    # Get label -> database mapping
    df_metadata = pd.read_csv('M:/ferrero/cytoimagenet/metadata.csv')
    to_database = df_metadata.groupby(by=['label']).apply(
        lambda x: tuple(sorted(x['database'].unique()))).reset_index()
    label_mapping_database = dict(
        zip(to_database.iloc[:, 0].tolist(), to_database.iloc[:, 1]))

    # df_diversity['database'] = df_diversity.label.map(lambda x: label_mapping_database[x])
