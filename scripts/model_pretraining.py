from prepare_dataset import check_exists, construct_cytoimagenet
from preprocessor import normalize

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential, initializers
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam

import os
import pandas as pd
import matplotlib.pyplot as plt

# PATHS
annotations_dir = "/home/stan/cytoimagenet/annotations/"
model_dir = "/home/stan/cytoimagenet/model/"
plot_dir = "/home/stan/cytoimagenet/figures/training/"
cyto_dir = '/ferrero/cytoimagenet/'


def preprocess_input(x):
    if x.max() == 255 and x.min == 0:
        return x / 255
    else:
        return normalize(x) / 255


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


def plot_loss(history, weight):
    """Plot training and validation results over the number of epochs."""
    plt.plot(history.history['loss'], label='Training Set')
    plt.plot(history.history['val_loss'], label='Validation Data)')
    plt.ylabel('Categorical Cross Entropy Loss')
    plt.xlabel('Num Epochs')
    plt.title('Performance on Training vs. Validation')
    plt.legend(loc="upper left")

    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    if weight is None:
        plt.savefig(f"{plot_dir}loss (random).png")
    else:
        plt.savefig(f"{plot_dir}loss (imagenet).png")


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

    for label in classes:
        df = pd.read_csv(f"{annotations_dir}classes/{label}.csv")
        # Check if images exist. If not, create images.
        exists_series = df.apply(check_exists, axis=1)

        if not all(exists_series):
            df_valid = df[exists_series]
            if len(df_valid) >= 287:
                print(f"label {label} now has {len(df_valid)} images!")
                df_valid.to_csv(f"{annotations_dir}classes/{label}.csv", index=False)

    construct_cytoimagenet(classes)
    print("Successfully Constructed CytoImageNet!")

    num_classes = len(classes)

    # Parameters
    batch_size = 64
    num_epochs = 10
    learn_rate = 1e-2
    weights = 'imagenet'                  # None or 'imagenet'

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
    adam = Adam(lr=learn_rate)

    # Compile
    model.compile(adam, loss="categorical_crossentropy",
                  metrics=["accuracy"])

    # Fit model
    history = model.fit(ds_train,
                        steps_per_epoch=steps_per_epoch_train,
                        validation_data=ds_val,
                        validation_steps=steps_per_epoch_val,
                        epochs=num_epochs,
                        verbose=1,
                        workers=8,
                        use_multiprocessing=True
                        )

    # Create plot
    plot_loss(history, weights)

    # Save model weights
    if not os.path.exists(f"{model_dir}cytoimagenet"):
        os.mkdir(model_dir + "cytoimagenet")

    if weights is None:
        model.save_weights(f"{model_dir}cytoimagenet/efficientnetb0_from_random.h5")
    else:
        model.save_weights(f"{model_dir}cytoimagenet/efficientnetb0_from_imagenet.h5")







