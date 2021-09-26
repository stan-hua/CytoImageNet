import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
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

# ==Label-related Helper Functions==:
def dirlabel_to_label(labels):
    """HELPER FUNCTION. Converts directory labels to original labels if
    possible."""
    df = pd.read_csv("/ferrero/cytoimagenet/metadata.csv")
    fixed_labels = []
    present_labels = df.label.unique()
    for label in labels:
        if label in present_labels:
            fixed_labels.append(label)
        else:
            fixed_labels.append(
                df[df.filename.str.contains(label)].iloc[0].label)
    return fixed_labels


def get_labels(diverse: bool = False):
    """If not <diverse>, return list of all labels.

    If <diverse>, return list of labels whose mean pairwise INTER cosine
    distance is greater than 0.8.

    NOTE: This removes 341 labels. Most of which are Recursion compound labels.
    """
    if not diverse:
        df = pd.read_csv('/ferrero/cytoimagenet/metadata.csv')
    else:
        df_diversity = pd.read_csv(f'{model_dir}similarity/full_diversity(cytoimagenet).csv')
        # Apply threshold
        df_diversity = df_diversity[df_diversity.inter_cos_distance_MEAN > 0.8]
        return df_diversity.label.tolist()


# ==Data Loading==:
def load_dataset(batch_size: int = 64, split=False, labels=None):
    """Return tuple of (training, validation) data iterators, constructed from
    metadata.

    With following fixed parameters:
        - target size: (224, 224)
        - color mode: RGB
        - shuffle: True
        - seed: 7779836983
        - interpolation: bilinear

    TODO: Try Augmentations
    # brightness_range=[0.6, 0.9]
    # zoom_range=[0.5, 1.5]
    """
    if labels is None:
        if split:  # If train-val split
            datagen = ImageDataGenerator(validation_split=0.1,
                                         rotation_range=360,
                                         fill_mode='reflect')
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
        else: # If don't use train-val split
            datagen = ImageDataGenerator(rotation_range=360, fill_mode='reflect')
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

    # Specify to use predefined labels.
    elif labels == 'toy_20':
        labels = toy_20
    elif labels == 'toy_50':
        labels = toy_50

    # Use metadata to create generators of image batches
    df = pd.read_csv('/ferrero/cytoimagenet/metadata.csv')
    df = df[df.label.isin(labels)]
    df['full_path'] = df.apply(lambda x: x.path + "/" + x.filename, axis=1)

    if split:   # if train-val split
        df_train, df_val = train_test_split(df, test_size=0.1, random_state=0,
                                            stratify=df['label'])
        train_gen = ImageDataGenerator(
            rotation_range=360, fill_mode='reflect',
            horizontal_flip=True,
        )
        train_generator = train_gen.flow_from_dataframe(
            dataframe=df_train,
            directory=None,
            x_col='full_path',
            y_col='label',
            batch_size=batch_size,
            target_size=(224, 224),
            interpolation="bilinear",
            color_mode="rgb",
            shuffle=True,
            seed=728565,
        )
        val_gen = ImageDataGenerator()
        val_generator = val_gen.flow_from_dataframe(
            dataframe=df_val,
            directory=None,
            x_col='full_path',
            y_col='label',
            shuffle=False,
            batch_size=batch_size,
            target_size=(224, 224),
            interpolation="bilinear",
            color_mode="rgb",
            seed=728565
        )
        return train_generator, val_generator
    else:
        # If no train-val split
        datagen = ImageDataGenerator(
            rotation_range=360, fill_mode='reflect'
        )
        train_generator = datagen.flow_from_dataframe(
            dataframe=df,
            directory=None,
            x_col='full_path',
            y_col='label',
            batch_size=batch_size,
            target_size=(224, 224),
            interpolation="bilinear",
            color_mode="rgb",
            shuffle=True,
            seed=728565
        )
        return train_generator, None


def get_dset_generators(split=False, num_classes=894, batch_size=64,
                        labels=None) -> tuple:
    """Return tuple of dataset generators. If split == False, return tuple with
    only 1 training generator. Else, return tuple with training and validation
    data generator.
    """
    # Get data generators
    train_gen, val_gen = load_dataset(batch_size=batch_size, split=split,
                                      labels=labels)
    # Number of Steps
    steps_per_epoch_train = train_gen.n // train_gen.batch_size
    # Create tf.data.Dataset objects from generator
    ds_train = tf.data.Dataset.from_generator(
        lambda: train_gen,
        output_types=(tf.float32, tf.float32),
        output_shapes=([None, 224, 224, 3], [None, num_classes]))
    # Prefetch data
    ds_train = ds_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    if split:  # If split into train-val sets
        steps_per_epoch_val = val_gen.n // val_gen.batch_size
        ds_val = tf.data.Dataset.from_generator(
            lambda: val_gen,
            output_types=(tf.float32, tf.float32),
            output_shapes=([None, 224, 224, 3], [None, num_classes]))
        ds_val = ds_val.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return (ds_train, ds_val), (steps_per_epoch_train, steps_per_epoch_val)
    return (ds_train, None), (steps_per_epoch_train, None)


# ==Defining Model==:
def create_model(num_classes: int, weights=None, pooling="avg"):
    """Construct tensorflow model."""
    if pooling == "max":
        efficient_model = EfficientNetB0(weights=weights,
                                         include_top=False,
                                         input_shape=(224, 224, 3),
                                         pooling="max")
        efficient_model.trainable = True

        # Prediction layers
        x = Dropout(0.2)(efficient_model.output)
        x = Dense(num_classes, activation=None)(x)

        # Convert to float32 values
        model_output = Activation('softmax', dtype='float32')(x)
        model = tf.keras.Model(efficient_model.input, model_output)
    else:
        if weights is None:
            model = EfficientNetB0(weights=weights,
                                   input_shape=(224, 224, 3),
                                   classes=num_classes)
            model.trainable = True
        elif weights == "imagenet":
            efficient_model = EfficientNetB0(weights=weights,
                                             input_shape=(224, 224, 3))
            efficient_model.trainable = True

            x = Dense(num_classes, activation='softmax')(efficient_model.layers[-2].output)
            model = tf.keras.Model(efficient_model.input, x)

    return model


# ==Plot==:
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
        ax2.plot(history.history['val_categorical_accuracy'],
                 label='Validation Data)')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Num Epochs')

    # Legend
    ax1.legend(loc='upper right')

    # Save Plot
    if weight is None:
        fig.savefig(f"{plot_dir}history (random, {dset} dataset).png")
    else:
        fig.savefig(f"{plot_dir}history (imagenet, {dset} dataset).png")


# ==Model Training==:
def main():
    global toy_50, toy_20
    # Dataset Parameters
    split = True
    classes = None                  # None or 'toy_20' or 'toy_50' or get_diverse_labels()
    num_classes = 894               # 894 or len(classes)
    dset = "full"          # plot label, 'full' or 'toy20' or 'toy50'
    save_weights_suffix = '(lr_0001_bs_64_epochs_100)'

    # Hyperparameters
    batch_size = 64
    num_epochs = 60
    learn_rate = 0.001
    weights = None              # initialize from random or 'imagenet'
    pooling = "avg"             # 'avg' or 'max'

    # Get data generators
    gens, steps = get_dset_generators(split=split, num_classes=num_classes,
                                      batch_size=batch_size, labels=classes)
    ds_train, ds_val = gens
    steps_per_epoch_train, steps_per_epoch_val = steps

    # Construct model
    model = create_model(num_classes, weights=weights, pooling=pooling)

    # Optimizer
    optimizer = Adam(lr=learn_rate)
    # optimizer = RMSprop(lr=learn_rate,
    #                     decay=0.9,
    #                     momentum=0.9)

    # Metrics
    top_n_acc = tf.keras.metrics.TopKCategoricalAccuracy(
        k=7, name='top_k_categorical_accuracy', dtype=None)

    # Compile
    model.compile(optimizer, loss="categorical_crossentropy",
                  metrics=['categorical_accuracy', top_n_acc])

    # Callbacks. Save model weights every 2 epochs.
    if weights is None:
        checkpoint_dir = model_dir + f"/cytoimagenet-weights/random_init/{dset}/"
        checkpoint_filename = checkpoint_dir + "efficientnetb0_from_random-epoch_{epoch:02d}.h5"
    else:
        checkpoint_dir = model_dir + f"/cytoimagenet-weights/imagenet_init/{dset}/"
        checkpoint_filename = checkpoint_dir + "efficientnetb0_from_imagenet_{epoch:02d}.h5"
    # Check if model weights directory exists. If not, create
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    callbacks = [ModelCheckpoint(checkpoint_filename, monitor='loss', verbose=0,
                                 save_best_only=False, save_weights_only=True,
                                 save_freq='epoch')]
                 # ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5,
                 #                   verbose=1,
                 #                   min_lr=0.001)]

    # Fit model
    if split:
        history = model.fit(ds_train,
                            steps_per_epoch=steps_per_epoch_train,
                            validation_data=ds_val,
                            validation_steps=steps_per_epoch_val,
                            epochs=num_epochs,
                            callbacks=callbacks,
                            verbose=1,
                            use_multiprocessing=False
                            )
    else:
        history = model.fit(ds_train,
                            steps_per_epoch=steps_per_epoch_train,
                            epochs=num_epochs,
                            callbacks=callbacks,
                            verbose=1,
                            use_multiprocessing=False
                            )

    # Create plot
    plot_loss(history, weights, dset, split)

    # Save model weights & history
    if weights is None:
        save_dir = f"{model_dir}cytoimagenet-weights/random_init/{dset}/"
    else:
        save_dir = f"{model_dir}cytoimagenet-weights/imagenet_init/{dset}/"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # Save training history
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(
        f"{save_dir}/history{save_weights_suffix}.csv", index=False)
    # Save weights
    model.save_weights(
        f"{save_dir}/efficientnetb0_from_random{save_weights_suffix}.h5")


if __name__ == "__main__":
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

    # toy_50 = dirlabel_to_label(toy_50)
    # toy_20 = dirlabel_to_label(toy_20)

    # Run main
    main()
