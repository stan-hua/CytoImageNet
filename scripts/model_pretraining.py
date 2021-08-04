from prepare_dataset import check_exists, construct_cytoimagenet
from preprocessor import normalize

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential, initializers
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob


# PATHS
annotations_dir = "/home/stan/cytoimagenet/annotations/"
model_dir = "/home/stan/cytoimagenet/model/"
plot_dir = "/home/stan/cytoimagenet/figures/training/"
cyto_dir = '/ferrero/cytoimagenet/'


# Only use CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def preprocess_input(x):
    if x.max() == 255.0 and x.min() == 0.0:
        return x / 255.
    elif x.max() == 0:
        print("Max is 0")
        print(x.shape)
        print(x.max(), x.min())
        return x / 255.
    elif x.max() == x.min():        # image improperly read by tensorflow
        print("Improperly Read!")
        print(x.shape)
        print(x.max(), x.min())
        return x / 255.
    else:
        # Get 0.1 and 99.9th percentile of pixel intensities
        top_001 = np.percentile(x.flatten(), 99.9)
        bot_001 = np.percentile(x.flatten(), 0.1)

        # Limit maximum intensity to 0.99th percentile
        x[x > top_001] = top_001

        # Then subtract by the 0.1th percentile intensity
        x = x - bot_001

        # Round intensities below 0.1th percentile to 0.
        x[x < 0] = 0
        if x.max() == 0:
            print("Divide by Zero")
            return x / 255.

        # Normalize between 0 and 1
        return x / x.max()
        # img = np.stack([x] * 3, axis=-1)
        # return img


def load_dataset(batch_size: int = 256):
    """Return tuple of (training, validation) data iterators, constructed from
    directory structure at 'ferrero/cytoimagenet/'

    With following parameters:
        - batch size: 256
        - target size: (224, 224)
        - color mode: RGB
        - shuffle: True
        - seed: 7779836983

    """
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                 validation_split=0.2)
    # Try
    # brightness_range=[0.6, 0.9]
    # zoom_range=[0.5, 1.5]
    # rotation_range = 30

    train_generator = datagen.flow_from_directory(
        directory=cyto_dir,
        batch_size=batch_size,
        target_size=(224, 224),
        interpolation="bilinear",
        color_mode="rgb",
        shuffle=True,
        seed=728565,
        subset="training"
    )

    val_generator = datagen.flow_from_directory(
        directory=cyto_dir,
        batch_size=batch_size,
        target_size=(224, 224),
        interpolation="bilinear",
        color_mode="rgb",
        shuffle=True,
        seed=728565,
        subset="validation"
    )
    return train_generator, val_generator


def create_model(num_classes: int, weights=None):
    """Construct tensorflow model."""
    model = Sequential()

    # Initialize EfficientNetB0
    base_model = EfficientNetB0(weights=weights,
                                include_top=False,
                                input_shape=(224, 224, 3),
                                pooling="max")
    base_model.trainable = True

    # Add efficientnet architecture
    model.add(base_model)

    # Prediction layer
    model.add(Dense(num_classes, activation="softmax"))
    return model


def plot_loss(history, weight, dset: str = "toy"):
    """Plot training and validation results over the number of epochs."""
    # Check if parent directory exists
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    # Plot Loss
    ax1.plot(history.history['loss'], label='Training Set')
    ax1.plot(history.history['val_loss'], label='Validation Data)')
    ax1.set_ylabel('Categorical Cross Entropy Loss')
    # ax1.set_title('Training vs. Validation Loss')

    # Plot Accuracy
    ax2.plot(history.history['categorical_accuracy'], label='Training Set')
    ax2.plot(history.history['val_categorical_accuracy'], label='Validation Data)')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Num Epochs')
    # ax2.set_title('Training vs. Validation Accuracy')

    # Legend
    try:
        plt.figlegend()
    except:
        plt.legend()

    # Save Plot
    if weight is None:
        fig.savefig(f"{plot_dir}history (random, {dset} dataset).png")
    else:
        fig.savefig(f"{plot_dir}history (imagenet, {dset} dataset).png")


def main():
    classes = [i.split("classes/")[-1].split(".csv")[0] for i in glob.glob(annotations_dir + "classes/*.csv")]
    num_classes = len(classes)

    # Parameters
    batch_size = 64
    num_epochs = 10
    learn_rate = 1e-3
    weights = None                  # None or 'imagenet'
    dset = "full"

    # Get data generators
    train_gen, val_gen = load_dataset(batch_size=batch_size)

    # Number of Steps
    steps_per_epoch_train = train_gen.n//train_gen.batch_size
    steps_per_epoch_val = val_gen.n//val_gen.batch_size

    # Create tf.data.Dataset objects from generator
    ds_train = tf.data.Dataset.from_generator(
        lambda: train_gen,
        output_signature=(
            tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)
        ))

    ds_val = tf.data.Dataset.from_generator(
        lambda: val_gen,
        output_signature=(
            tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)
        ))

    # Construct model
    model = create_model(num_classes, weights=weights)

    # Optimizer
    adam = Adam(lr=learn_rate,
                clipnorm=1.0)

    # Compile
    model.compile(adam, loss="categorical_crossentropy",
                  metrics=['categorical_accuracy'])

    # Callbacks. Save model weights every 2 epochs.
    if weights is None:
        checkpoint_filename = model_dir + "/cytoimagenet-weights/random_init/efficientnetb0_from_random-epoch_{epoch:02d}.h5"
    else:
        checkpoint_filename = model_dir + "/cytoimagenet-weights/imagenet_init/efficientnetb0_from_imagenet_{epoch:02d}.h5"
    callbacks = [tf.keras.callbacks.ModelCheckpoint(
        checkpoint_filename, monitor='loss', verbose=0,
        save_best_only=False, save_weights_only=True, save_freq=2*batch_size)]

    # Fit model
    history = model.fit(ds_train,
                        steps_per_epoch=steps_per_epoch_train,
                        validation_data=ds_val,
                        validation_steps=steps_per_epoch_val,
                        epochs=num_epochs,
                        verbose=1,
                        workers=8,
                        use_multiprocessing=True,
                        callbacks=callbacks
                        )

    # Create plot
    plot_loss(history, weights, dset)

    # Save model weights
    if not os.path.exists(f"{model_dir}cytoimagenet"):
        os.mkdir(model_dir + "cytoimagenet")

    if weights is None:
        model.save_weights(f"{model_dir}cytoimagenet/efficientnetb0_from_random.h5")
    else:
        model.save_weights(f"{model_dir}cytoimagenet/efficientnetb0_from_imagenet.h5")


if __name__ == "__main__":
    # Labels
    classes = ['human', 'nucleus', 'cell membrane',
              'white blood cell', 'kinase',
              'wildtype', 'difficult',
              'nematode', 'yeast', 'bacteria',
              'vamp5 targeted', 'uv inactivated sars-cov-2',
              'trophozoite', 'tamoxifen', 'tankyrase inhibitor',
              'dmso', 'rho associated kinase inhibitor', 'rna',
              'rna synthesis inhibitor', 'cell body'
              ]

    # for label in classes:
    #     df = pd.read_csv(f"{annotations_dir}classes/{label}.csv")
    #     # Check if images exist. If not, create images.
    #     exists_series = df.apply(check_exists, axis=1)
    #
    #     if not all(exists_series):
    #         df_valid = df[exists_series]
    #         if len(df_valid) >= 287:
    #             print(f"label {label} now has {len(df_valid)} images!")
    #             df_valid.to_csv(f"{annotations_dir}classes/{label}.csv", index=False)

    # construct_cytoimagenet(classes, overwrite=False)
    # print("Successfully Constructed CytoImageNet!")

    pass
    df_metadata = pd.read_csv("/ferrero/cytoimagenet/metadata.csv")
    print(df_metadata[df_metadata.unreadable].apply(lambda x: x.path + "/" + x.filename, axis=1).tolist())
    print(len(df_metadata[df_metadata.unreadable]), " Unreadable Files!")
    #
    # x = df_metadata[df_metadata.idx == 'idr0003-2645']
    # print(x.path.iloc[0] + "/" + x.filename.iloc[0])
