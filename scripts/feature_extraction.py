from prepare_dataset import check_exists
from analyze_metadata import get_df_counts

import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
from itertools import combinations
import os

from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential, initializers
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Flatten, Dense, Dropout, Input
import tensorflow as tf

# Limit number of GPU used to 1
tf.config.set_soft_device_placement(True)
tf.debugging.set_log_device_placement(True)


model_dir = "/home/stan/cytoimagenet/model/"


def build_alexnet():
    """Build AlexNet with randomly initialized weights"""
    # Random Initializers
    weights_init = initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
    bias_init = 'zeros'

    # Build Model
    model = Sequential(name="AlexNet")
    model.add(Input(shape=(224, 224, 3)))

    # Conv Block
    model.add(Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu',
                     kernel_initializer=weights_init,
                     bias_initializer=bias_init
                     ))
    model.add(BatchNormalization())

    # Conv Block
    model.add(MaxPool2D(pool_size=(3,3), strides=(2,2)))
    model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same",
                     kernel_initializer=weights_init,
                     bias_initializer=bias_init
                     ))
    model.add(BatchNormalization())

    # Conv Block
    model.add(MaxPool2D(pool_size=(3,3), strides=(2,2)))
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same",
                     kernel_initializer=weights_init,
                     bias_initializer=bias_init
                     ))
    model.add(BatchNormalization())

    # Conv Block
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same",
                     kernel_initializer=weights_init,
                     bias_initializer=bias_init
                     ))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same",
                     kernel_initializer=weights_init,
                     bias_initializer=bias_init
                     ))
    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(3,3), strides=(2,2)))
    model.add(Flatten())

    # Fully Connected Block
    model.add(Dense(4096, activation='relu',
                    kernel_initializer=weights_init,
                    bias_initializer=bias_init
                    ))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu',
                    kernel_initializer=weights_init,
                    bias_initializer=bias_init
                    ))
    # model.add(Dropout(0.5))
    # model.add(Dense(10, activation='softmax',
    #                 kernel_initializer=weights_init,
    #                 bias_initializer=bias_init
    #                 ))
    return model


def test_datagen(label: str):
    """Return dataframe compatible with tf.keras
    ImageDataGenerator.flow_from_dataframe.
    """
    df = pd.read_csv(f"/home/stan/cytoimagenet/annotations/classes/{label}.csv")
    df.apply(check_exists, axis=1)
    def get_filename(x):
        """Returns filename that satisfies: '/ferrero/stan_data/' + filename."""
        return x.dir_name + f"{x.path}/{x.filename}".split(x.dir_name)[-1]

    df['full_name'] = df.apply(get_filename, axis=1)
    return df


def get_activations_for(model, label: str):
    """Load <model>. Extract activations for images under label.
    """
    # Create Image Generator for <label>
    test_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
        dataframe=test_datagen(label),
        directory='/ferrero/stan_data/',
        x_col='full_name',
        color_mode='rgb',
        batch_size=1,
        target_size=(224, 224),
        shuffle=False,
        class_mode=None,
        interpolation='bilinear',
        seed=1
    )

    # Save image embeddings
    activations = model.predict(test_gen)
    pd.DataFrame(activations).to_csv(f"{model_dir}random_model-activations/{label}_activations.csv", index=False)
    return activations


def batch_cos_sim(imgs_feature) -> list:
    """Return cosine similarities for all pairs of image features <imgs_feature>
    """
    pairs = combinations(imgs_feature, 2)

    cos_sims = []
    for pair in pairs:
        cos_sims.append(cosine_similarity(pair))
    return cos_sims


def cosine_similarity(img1_img2: tuple) -> float:
    """Return cosine similarity between image features of <img1_img2>[0] and
    <img1_img2>[1].
    """
    img1, img2 = img1_img2
    return dot(img1, img2) / (norm(img1) * norm(img2))


if __name__ == "__main__" and "D:\\" not in os.getcwd():
    labels = ["ifn-gamma",
             "factor-x",
             "nfkb pathway inhibitor",
             "s8645",
             "uv inactivated sars-cov-2"]

    labels_2 = ["cell membrane",
                "hpsi0813i-ffdm_3",
                "osteopontin",
                "pik3ca targeted",
                "bacteria",
                "voltage-gated sodium channel blocker",
                "s501357"
                ]


    # Build model.
    model = build_alexnet()
    # model.save_weights(f"{model_dir}random_alex.h5")
    model.load_weights(f"{model_dir}random_alex.h5")

    sim_label = labels.copy()
    sim_mean = []
    sim_std = []

    for label in labels_2:
        try:
            if os.path.exists(f"{model_dir}random_model-activations/{label}_activations.csv"):
                features = pd.read_csv(f"{model_dir}random_model-activations/{label}_activations.csv").to_numpy()
            else:
                features = get_activations_for(model, label)
            if os.path.exists(f"{model_dir}similarity/{label}_sim.npy"):
                cos_sims = np.load(f"{model_dir}similarity/{label}_sim.npy")
            else:
                print(f"Calculating Cosine Similarity for {label}...")
                cos_sims = np.array(batch_cos_sim(features))
                np.save(f"{model_dir}similarity/{label}_sim.npy", cos_sims)

            print(f"{label} Similarity | mean: {cos_sims.mean()}, sd: {cos_sims.std()}")
            print()

            sim_mean.append(cos_sims.flatten().mean())
            sim_std.append(cos_sims.flatten().std())
        except:
            print(label + " failed!")
            sim_label.remove(label)

    print(sim_label)
    print(sim_mean)
    print(sim_std)
    if os.path.exists(f"{model_dir}similarity/label_similarities.csv"):
        df_sim = pd.read_csv(f"{model_dir}similarity/label_similarities.csv")
        df_sim = pd.concat([pd.read_csv(f"{model_dir}similarity/label_similarities.csv"),
                           pd.DataFrame({"label": sim_label,
                                         "mean_sim": sim_mean,
                                         "std_sim": sim_std})]
                           )
    else:
        df_sim = pd.DataFrame({"label": sim_label,
                               "mean_sim": sim_mean,
                               "std_sim": sim_std})
    df_sim.to_csv(f"{model_dir}similarity/label_similarities.csv", index=False)
else:
    df_counts = get_df_counts()
    df_sim = pd.read_csv(f"{model_dir}similarity/label_similarities.csv")

    df_sim = pd.merge(df_sim, df_counts, how="left", on ="label")



label = "cell membrane"
df_a = pd.read_csv(f"M:/home/stan/cytoimagenet/annotations/classes/{label}.csv")
df_a.name.value_counts()
