import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Flatten, Dense, Dropout, Input, Activation
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam, RMSprop

from tensorflow.keras import mixed_precision

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
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# List GPU
print(tf.config.list_physical_devices('GPU'))

# Use Mixed Precision
# from tensorflow.keras.mixed_precision import experimental as mixed_precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)


def load_dataset(batch_size: int = 64, split=False):
    """Return tuple of (training, validation) data iterators, constructed from
    directory structure at 'ferrero/cytoimagenet/'

    With following parameters:
        - batch size: 64
        - target size: (224, 224)
        - color mode: RGB
        - shuffle: True
        - seed: 7779836983
    Try Augmentations
    # brightness_range=[0.6, 0.9]
    # zoom_range=[0.5, 1.5]
    # rotation_range = 30
    """
    # If train-val split
    if split:
        datagen = ImageDataGenerator(validation_split=0.2)
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
    # If don't use train-val split
    datagen = ImageDataGenerator(rescale=1/255.)
    train_generator = datagen.flow_from_directory(
        directory=cyto_dir,
        batch_size=batch_size,
        target_size=(224, 224),
        interpolation="bilinear",
        color_mode="rgb",
        shuffle=True,
        seed=728565
    )
    return train_generator, None


def get_dset_generators(split=False, num_classes=902, batch_size=64) -> tuple:
    """Return tuple of dataset generators. If split == False, return tuple with
    only 1 training generator. Else, return tuple with training and validation
    data generator.
    """
    # Get data generators
    train_gen, val_gen = load_dataset(batch_size=batch_size, split=split)
    # Number of Steps
    steps_per_epoch_train = train_gen.n//train_gen.batch_size
    # Create tf.data.Dataset objects from generator
    ds_train = tf.data.Dataset.from_generator(
        lambda: train_gen,
        output_types=(tf.float32, tf.float32),
        output_shapes=([None, 224, 224, 3], [None, num_classes]))
    if split:   # If split into train-val sets
        # Number of Steps
        steps_per_epoch_val = val_gen.n//val_gen.batch_size

        # Create tf.data.Dataset objects from generator
        ds_val = tf.data.Dataset.from_generator(
            lambda: val_gen,
            output_types=(tf.float32, tf.float32),
            output_shapes=([None, 224, 224, 3], [None, num_classes]))
        return (ds_train, ds_val), (steps_per_epoch_train, steps_per_epoch_val)
    return (ds_train, None), (steps_per_epoch_train, None)


def create_model(num_classes: int, weights=None):
    """Construct tensorflow model."""
    efficient_model = EfficientNetB0(weights=weights,
                                     include_top=False,
                                     input_shape=(224, 224, 3),
                                     pooling="max")
    efficient_model.trainable = True
    # Prediction layers
    x = Dense(num_classes, activation=None)(efficient_model.output)
    # Convert to float32 values
    model_output = Activation('softmax', dtype='float32')(x)

    return tf.keras.Model(efficient_model.input, model_output)


def plot_loss(history, weight, dset: str = "toy", split=False):
    """Plot training and validation results over the number of epochs."""
    # Check if parent directory exists
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    # Plot Loss
    ax1.plot(history.history['loss'], label='Training Set')
    if split:
        ax1.plot(history.history['val_loss'], label='Validation Data)')
    ax1.set_ylabel('Categorical Cross Entropy Loss')

    # Plot Accuracy
    ax2.plot(history.history['categorical_accuracy'], label='Training Set')
    if split:
        ax2.plot(history.history['val_categorical_accuracy'], label='Validation Data)')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Num Epochs')

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
    num_classes = 901

    # Parameters
    batch_size = 32
    num_epochs = 10
    learn_rate = 0.01
    weights = None                              # None or 'imagenet'
    dset = "full"                               # 'full' or 'toy'
    split = False

    # Get data generators
    gens, steps = get_dset_generators(split=split, num_classes=num_classes,
                                      batch_size=batch_size)
    ds_train, ds_val = gens
    steps_per_epoch_train, steps_per_epoch_val = steps

    # Construct model
    model = create_model(num_classes, weights=weights)

    # Optimizer
    optimizer = Adam(lr=learn_rate)
    # optimizer = RMSprop(lr=learn_rate,
    #                     decay=0.9,
    #                     momentum=0.9)

    # Compile
    model.compile(optimizer, loss="categorical_crossentropy",
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
    if split:
        history = model.fit(ds_train,
                            steps_per_epoch=steps_per_epoch_train,
                            validation_data=ds_val,
                            validation_steps=steps_per_epoch_val,
                            epochs=num_epochs,
                            verbose=1,
                            callbacks=callbacks,
                            workers=8,
                            use_multiprocessing=True
                            )
    else:
        history = model.fit(ds_train,
                            steps_per_epoch=steps_per_epoch_train,
                            epochs=num_epochs,
                            verbose=1,
                            callbacks=callbacks,
                            workers=8,
                            use_multiprocessing=True
                            )

    # Create plot
    plot_loss(history, weights, dset, split)

    # Save model weights
    if not os.path.exists(f"{model_dir}cytoimagenet"):
        os.mkdir(model_dir + "cytoimagenet")

    if weights is None:
        model.save_weights(f"{model_dir}cytoimagenet/efficientnetb0_from_random.h5")
    else:
        model.save_weights(f"{model_dir}cytoimagenet/efficientnetb0_from_imagenet.h5")


if __name__ == "__main__":
    # Labels
    # classes = ['human', 'nucleus', 'cell membrane',
    #           'white blood cell', 'kinase',
    #           'wildtype', 'difficult',
    #           'nematode', 'yeast', 'bacteria',
    #           'vamp5 targeted', 'uv inactivated sars-cov-2',
    #           'trophozoite', 'tamoxifen', 'tankyrase inhibitor',
    #           'dmso', 'rho associated kinase inhibitor', 'rna',
    #           'rna synthesis inhibitor', 'cell body'
    #           ]
    main()
