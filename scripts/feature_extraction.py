import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import EfficientNetB0

import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm

from sklearn import preprocessing
from sklearn.decomposition import PCA

from itertools import combinations
import datetime
import multiprocessing
import os
import random
import glob

import cv2
import seaborn as sns
import matplotlib.pyplot as plt


# Set CPU only
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Plot Settings
sns.set_style("dark")
plt.style.use('dark_background')


# Global Variables
if "D:\\" not in os.getcwd():
    annotations_dir = "/home/stan/cytoimagenet/annotations/"
    data_dir = '/ferrero/stan_data/'
    model_dir = "/home/stan/cytoimagenet/model/"
    plot_dir = "/home/stan/cytoimagenet/figures/"
    cyto_dir = '/ferrero/cytoimagenet/'
else:
    annotations_dir = "M:/home/stan/cytoimagenet/annotations/"
    data_dir = 'M:/ferrero/stan_data/'
    model_dir = "M:/home/stan/cytoimagenet/model/"
    plot_dir = "M:/home/stan/cytoimagenet/figures/"
    cyto_dir = 'M:/ferrero/cytoimagenet/'

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
    time_delta = (end - start)
    total_seconds = time_delta.total_seconds()
    minutes = total_seconds/60
    print("Finished in ", round(minutes, 2), " minutes!")
    return minutes


def test_datagen(label: str, label_dir, data_subset="train"):
    """Return dataframe compatible with ImageDataGenerator.flow_from_dataframe
    from tensorflow keras.

    Since API is only compatible with jpg, png and gif files. Check that all files

    """
    if data_subset != 'val':
        if label == 'toy_20':      # use labels from toy_20
            accum_df = []
            for l in toy_20:
                curr_df = pd.read_csv(f"{annotations_dir}classes/{label_dir}/{l}.csv")
                # Check if exists
                exists = curr_df.apply(lambda x: os.path.exists(x.path + "/" + x.filename), axis=1)
                curr_df = curr_df[exists]
                print(len(curr_df), ' exist for ', l)
                # Sample 100
                accum_df.append(curr_df.sample(n=100))
            df = pd.concat(accum_df, ignore_index=True)
        else:
            df = pd.read_csv(f"{annotations_dir}classes/{label_dir}/{label}.csv")
    else:
        df = pd.read_csv("/ferrero/stan_data/unseen_classes/metadata.csv")
    df['full_name'] = df.apply(lambda x: f"{x.path}/{x.filename}", axis=1)
    return df


def load_img(x):
    """Load image using Open-CV. Resize to 224, 224"""
    img = cv2.imread(x)
    if img is None:
        return
    img_resized = cv2.resize(img, (224, 224), cv2.INTER_LINEAR)
    print("Shape: ", img_resized.shape)
    return img_resized


def get_activations_for(model, label: str, directory: str = "imagenet-activations/base",
                        label_dir: str = "", method='gen', data_subset="train"):
    """Load <model>. Extract activations for images under label.
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
    else:   # iterative method
        df_test = test_datagen(label, label_dir, data_subset)
        activations = []

        pool = multiprocessing.Pool(30)
        try:
            imgs = pool.map(load_img, df_test.full_name.tolist())
            pool.close()
            pool.join()
        except:
            pool.close()
        idx_none = []
        final_imgs = []
        for i in range(len(imgs)):
            if imgs[i] is None:
                idx_none.append(i)
            else:
                final_imgs.append(imgs[i])
        print(f"Found {len(idx_none)} Nones! ", idx_none)
        final_imgs = np.array(final_imgs)
        print("Final Shape: ", final_imgs.shape)
        activations = model.predict(final_imgs)

    # Save activations
    if not os.path.exists(f"{model_dir}{directory}"):
        os.mkdir(f"{model_dir}{directory}")
    if data_subset != 'val':
        pd.DataFrame(activations).to_csv(f"{model_dir}{directory}/{label}_activations.csv", index=False)
    else:
        pd.DataFrame(activations).to_csv(f"{model_dir}{directory}/unseen_classes_embeddings ({weights}, {dset}).csv", index=False)
    return activations


# DATA GENERATOR
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
        batch_size=1,
        target_size=(224, 224),
        interpolation="bilinear",
        class_mode=None,
        color_mode="rgb",
        shuffle=True,
        seed=728565
    )
    return train_generator, df.label.tolist()


def extract_embeds_from_sample(dset, weights='/home/stan/cytoimagenet/model/cytoimagenet-weights/random_init/toy_20/efficientnetb0_from_random(lr_0001_bs_64)-notop.h5'):
    """Sample 100 images from each class. Extract embeddings for all samples and
    save in {model_dir}/imagenet-activations/full_dset_embeddings.csv"""
    global model_dir

    # Initialize pretrained EfficientNetB0.
    model = EfficientNetB0(weights=weights,
                           include_top=False,
                           input_shape=(224, 224, 3),
                           pooling="avg")
    # Load sampled images
    datagen, labels = load_dataset(dset)
    ds_test = tf.data.Dataset.from_generator(lambda: datagen, output_types=(tf.float32))
    ds_test = ds_test.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # Number of Steps till done Predicting
    steps_to_predict = datagen.n // datagen.batch_size
    print("Beginning feature extraction...")
    embeds = model.predict(ds_test, batch_size=1, steps=steps_to_predict,
                           use_multiprocessing=True, workers=32, verbose=1)
    print("Done extracting!")
    sampled_embeds = pd.DataFrame(embeds)
    print('Saving features...')
    sampled_embeds.to_hdf(f"{model_dir}/imagenet-activations/{dset}_dset_embeddings.h5", key='embed', index=False)
    pd.Series(labels).to_hdf(f"{model_dir}/imagenet-activations/{dset}_dset_embeddings.h5", key='label', index=False)
    print("Done saving features!")


# SIMILARITY METRICS
def intra_similarity(label_features: np.array, n=10000) -> list:
    """Calculate intra cosine similarities between images images (features) from one
    label. Estimate by calculating cosine similarity for at most <n> pairs.

    Return list of pairwise cosine similarities.

    ==Parameters==:
        label_features: np.array where each row is a feature vector for one
                        image
    """
    # ==REDUCE DIMENSIONALITY==:
    # Normalize features between [0, 1]
    # label_features = preprocessing.normalize(label_features)

    # Reduce Dimensionality of Treatment Activations
    # pca = PCA()
    # proc_label_features = pca.fit_transform(label_features)
    # # Select only >= 99% cumulative percent variance
    # cum_percent_variance = pca.explained_variance_ratio_.cumsum()
    # num_pc = np.where(cum_percent_variance >= 0.999)[0][0]
    # print(num_pc, f'of {label_features.shape[1]} PCs Selected!')
    # label_features = proc_label_features[:, : num_pc+1]

    pairs = np.array(list(combinations(label_features, 2)))
    if len(pairs) > n:
        pairs = pairs[np.random.choice(len(pairs), n, replace=False)]

    cos_sims = []
    for pair in pairs:
        cos_sims.append(cosine_similarity(pair))
    return cos_sims


def inter_similarity(label_features: np.array, other_features: np.array,
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
        cos_sims.append(cosine_similarity((random.choice(label_features), random.choice(other_features))))
    return cos_sims


def cosine_similarity(img1_img2: tuple) -> float:
    """Return cosine similarity between image features of <img1_img2>[0] and
    <img1_img2>[1].
    """
    img1, img2 = img1_img2
    return dot(img1, img2) / (norm(img1) * norm(img2))


def calculate_diversity(label: str, dset='toy_20', plot=False):
    """Calculate inter-class diversity and intra-class diversity using pairwise
    cosine similarity. Return mean and SD for inter and intra-class diversity.

    ==Parameters==:
        label: CytoImageNet class to calculate diversity for
    """
    global model_dir

    # Verify that embeddings for samples of the full dataset exists
    if not os.path.exists(f"{model_dir}/imagenet-activations/{dset}_dset_embeddings.h5"):
        extract_embeds_from_sample(dset)

    df_embeds = pd.read_hdf(f"{model_dir}/cytoimagenet-activations/{dset}_dset_embeddings.h5", 'embed')
    labels = pd.read_hdf(f"{model_dir}/cytoimagenet-activations/{dset}_dset_embeddings.h5", 'label')
    df_embeds['label'] = labels

    # ==INTRA DIVERSITY==:
    # Calculate mean pairwise cosine similarity among 100 samples of the label
    label_vecs = df_embeds[df_embeds.label == label].drop(columns=['label']).to_numpy()
    other_label_vecs = df_embeds[df_embeds.label != label].drop(columns=['label']).to_numpy()
    intra_similarities = intra_similarity(label_vecs)

    # ==INTER DIVERSITY==:
    # Calculate mean pairwise cosine similarity between representative feature vector of label and other labels
    inter_similarities = inter_similarity(label_vecs, other_label_vecs)

    if np.random.choice(range(10), 1)[0] == 1 or plot:
        fig, (ax1, ax2) = plt.subplots(2, 1)
        sns.kdeplot(x=intra_similarities, hue=['Intra Diversity'] * len(intra_similarities), ax=ax1)
        sns.kdeplot(x=inter_similarities, hue=['Inter Diversity'] * len(inter_similarities), ax=ax2)
        ax1.set_xlabel("")
        ax1.set_xlim(-0.05, 1)
        ax2.set_xlabel("")
        ax2.set_xlim(-0.05, 1)
        plt.savefig(f"{plot_dir}classes/similarity/base_{label}_diversity.png")

    return (np.mean(intra_similarities), np.std(intra_similarities)), (np.mean(inter_similarities), np.std(inter_similarities))


def get_div(label, dset='toy_20'):
    (intra_mean, intra_sd), (inter_mean, inter_sd) = calculate_diversity(label, dset=dset)
    curr_div = pd.DataFrame({"label": label, "intra_cos_distance_MEAN": 1-intra_mean,
                             "intra_SD": intra_sd, "inter_cos_distance_MEAN": 1-inter_mean,
                             "inter_SD": inter_sd}, index=[0])

    return curr_div


def get_all_diversity(dset='full', kind=''):
    """Save dataframe of mean inter cosine similarity and mean intra
    cosine similarity for all 894 CytoImageNet labels.
    """
    global toy_50, toy_20
    df_metadata = pd.read_csv("/ferrero/cytoimagenet/metadata.csv")

    if dset == 'toy_50':
        df_metadata = df_metadata[df_metadata.label.isin(toy_50)]
    elif dset == 'toy_20':
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
            accum_div.append(get_div(label, dset=dset))
        except:
            print(label, ' errored!')

    df_diversity = pd.concat(accum_div, ignore_index=True)
    df_diversity.to_csv(f'{model_dir}similarity/{kind}{dset}_diversity.csv', index=False)

    return df_diversity





def main():
    # Choose model
    weights = 'cytoimagenet'              # 'imagenet' or None
    cyto_weights_file = "efficientnetb0_from_random(lr_0001_bs_64).h5"
    dset = 'toy_20'
    data_subset = 'train'               # use unseen classes?
    num_classes = 20
    method_data_loading = 'iter'        # 'gen' vs 'iter'

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
            x = Dense(num_classes, activation="softmax")(efficientnet_model.output)
            old_model = tf.keras.Model(efficientnet_model.input, x)
            old_model.load_weights(f"{cyto_weights_dir}/{dset}/{cyto_weights_file}")
            # Remove prediction layer
            new_model = tf.keras.Model(old_model.input, old_model.layers[-2].output)
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

    get_activations_for(model, 'toy_20', activation_loc, method=method_data_loading,
                        data_subset=data_subset, label_dir=label_dir)


if __name__ == "__main__" and "D:\\" not in os.getcwd():
    # Get base diversity
    get_all_diversity('toy_20', kind='base_')
