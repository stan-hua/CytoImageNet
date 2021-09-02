from preprocessor import create_image, get_file_references, normalize

import glob
import os
import time
import random
import shutil

import pandas as pd
import numpy as np
from PIL import Image
import cv2

import tensorflow as tf

import multiprocessing
import PIL
from PIL import UnidentifiedImageError

# Only use CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# PATHS
if "D:\\" in os.getcwd():
    annotations_dir = "M:/home/stan/cytoimagenet/annotations/"
    plot_dir = "M:/home/stan/cytoimagenet/figures/classes/"
else:
    annotations_dir = "/home/stan/cytoimagenet/annotations/"
    plot_dir = "/home/stan/cytoimagenet/figures/classes/"


def check_exists(x):
    """Return True if image exists at <x>.

    If False, use dataset-specific method to create image. Raise Exception if
    image creation failed.
    """
    a = cv2.imread(x.path + "/" + x.filename, cv2.IMREAD_GRAYSCALE)
    if a is None or a.max() == 1:
        # print(f"Creating images for {x.dir_name}")
        # start = time.perf_counter()
        create_image(x)
        # print(f"Image Created in {time.perf_counter()-start} seconds!")
        if os.path.exists(x.path + "/" + x.filename):
            return True
        return False
    return True


def check_grayscale(x):
    """Return True if image is grayscale. Else, return False"""
    try:
        img = np.array(Image.open(x.path + "/" + x.filename))
    except:
        img = cv2.imread(x.path + "/" + x.filename, cv2.IMREAD_GRAYSCALE)

    if len(img.shape) == 2:
        return True
    else:
        return False


def check_readable(x):
    """Check if Image is Readable by Tensorflow.
    """
    try:
        image = PIL.Image.open(x.path + "/" + x.filename)
        return True
    except PIL.UnidentifiedImageError:
        print("PIL.UnidentifiedImageError!")
        return False
    # image = tf.io.read_file(x.path + "/" + x.filename)
    # image = tf.image.decode_png(image)
    # # image = tf.image.convert_image_dtype(image, tf.float32)
    # if (tf.reduce_max(image) == tf.reduce_min(image)) or image is None:         # Check if image is constant, or image is None
    #     return False
    # return True


def to_grayscale(x):
    """Convert RGB images to Grayscale when found."""
    try:
        img = np.array(Image.open(x.path + "/" + x.filename))
    except:
        img = cv2.imread(x.path + "/" + x.filename, cv2.IMREAD_GRAYSCALE)

    if len(img.shape) != 3:
        return

    # Normalize between 0 to 1 with respect to max of each channel
    for n in range(img.shape[2]):
        img[:, :, n] = normalize(img[:, :, n])

    # Average along channels. Then normalize between 0 to 255
    img = img.mean(axis=-1)
    img = img * 255

    Image.fromarray(img).convert("L").save(x.path + "/" + x.filename)
    print("Grayscale Conversion Successful!")


def to_png(x):
    # Skip if exists
    new_filename = ".".join(x.filename.split(".")[:-1]) + ".png"
    if os.path.exists(x.path + "/" + new_filename):
        return new_filename

    # Load Image
    img = cv2.imread(x.path + "/" + x.filename, cv2.IMREAD_GRAYSCALE)
    try:
        cv2.imwrite(x.path + "/" + new_filename, img)
        print("PNG Conversion Successful!")
        os.remove(x.path + "/" + x.filename)
        return new_filename
    except:
        print("FAILED! PNG Conversion")
        print(x.path + "/" + x.filename)
        print(os.path.exists(x.path + "/" + x.filename))
        print()
        return None


class Upsampler():
    def get_slicers(self, height, width, num_crops, min_height, min_width):
        """Return <num_crops> number of image slicers to crop image into <num_crops>
        based on exponentially decaying resolution. In the form (x_mins, x_maxs,
        y_mins, y_maxs), where each element is a parallel list with the others.


        NOTE: num_crops is at most 4.
        NOTE: x refers to width (or columns). y refers to height (or rows).
        """
        resolutions = [1/2, 1/4, 1/8, 1/16]

        # Check if all multipliers * (height, width) satisfy resolution
        to_remove = []
        for i in range(len(resolutions)):
            if resolutions[i] * height < min_height or resolutions[i] * width < min_width:
                to_remove.append(i)
        # Remove too small resolutions
        to_remove = [resolutions[i] for i in to_remove]
        for i in to_remove:
            resolutions.remove(i)

        # Check if # of resolutions is == num_crops.
        if len(resolutions) < num_crops:    # randomly duplicate to fill gap
            deficit = num_crops - len(resolutions)
            resolutions = np.append(resolutions, np.random.choice(resolutions, size=deficit, replace=True))
        elif len(resolutions) > num_crops:  # randomly choose resolutions to keep
            resolutions = np.random.choice(resolutions, size=num_crops)
        # Shuffle resolutions
        random.shuffle(resolutions)

        # Quadrant Indices & Match to number of crops
        quadrant_indices = []
        for i in [0, np.floor(width / 2)]:
            for j in [0, np.floor(height / 2)]:
                quadrant_indices.append((i, j))
        quadrant_indices = np.array(quadrant_indices)
        if len(quadrant_indices) > num_crops:
            quadrant_indices = quadrant_indices[np.random.choice(len(quadrant_indices), num_crops, replace=False)]

        n = 0
        x_mins = []
        x_maxs = []
        y_mins = []
        y_maxs = []
        for i, j in quadrant_indices:
            curr_width = np.floor(width * resolutions[n])
            curr_height = np.floor(height * resolutions[n])

            # If current resolutions are 1/2, don't change anchor point (x_min, x_max)
            if resolutions[n] == 1/2:
                x_min = i
                y_min = j
                x_max = i + curr_width
                y_max = j + curr_height
            # Else, randomly choose anchor point from available space
            else:
                x_interval = np.floor(width / 2) - curr_width
                y_interval = np.floor(height / 2) - curr_height

                x_min = i + random.randint(0, x_interval)
                x_max = x_min + curr_width

                y_min = j + random.randint(0, y_interval)
                y_max = y_min + curr_height

            # Make sure y_max and x_max are within the original image resolution
            x_max = min(x_max, width)
            y_max = min(y_max, height)

            # Update accumulators
            n += 1
            x_mins.append(int(x_min))
            x_maxs.append(int(x_max))
            y_mins.append(int(y_min))
            y_maxs.append(int(y_max))

        return (x_mins, x_maxs, y_mins, y_maxs), resolutions

    def create_crops(self, img: np.array, num_crops: int):
        """Return tuple of two tuples:
            - with n image crops of possible sizes (1/2, 1/4, 1/8, 1/16)
            - with cropping information (list of x_min, list of x_max,
                list of y_min, list of y_max)

        Precondition:
            - num_crops is at most 4.

        NOTE: If num_crops < 4, randomly select from slices.
        NOTE: If resolution does not reach threshold, randomly select form upper
            resolutions to assign to quadrant
        NOTE: The smallest acceptable crop is 70x70.
        """
        # Create Quadrants
        num_crops = min(num_crops, 4)       # enforce num_crops <= 4
        (x_mins, x_maxs, y_mins, y_maxs), scaling = self.get_slicers(
            img.shape[0], img.shape[1],
            num_crops,
            70, 70)

        # Accumulate image crops
        img_crops = []
        used_xmins = []
        used_xmaxs = []
        used_ymins = []
        used_ymaxs = []
        used_scaling = []
        # Loop through number of width crops & height crops
        for i in range(len(x_mins)):
            # img_crop = img.crop((x_mins[i], y_mins[i], x_maxs[i], y_maxs[i]))
            img_crop = img[y_mins[i]:y_maxs[i], x_mins[i]:x_maxs[i]]
            assert y_maxs[i] <= img.shape[0] and x_maxs[i] <= img.shape[1]
            # Save crop if mean pixel intensity is greater than 1 and less than 254. And 75th percentile is not 0.
            if img_crop.mean() > 1 and img_crop.mean() < 254 and np.percentile(img_crop, 75) != 0:
                img_crops.append(normalize(img_crop) * 255)
                used_xmins.append(x_mins[i])
                used_xmaxs.append(x_maxs[i])
                used_ymins.append(y_mins[i])
                used_ymaxs.append(y_maxs[i])
                used_scaling.append(scaling[i])

        return img_crops, (used_xmins, used_xmaxs, used_ymins, used_ymaxs, used_scaling)

    def save_crops(self, imgs: list, x) -> list:
        """Return list of new filenames for crops.

        Save image crops <imgs> in the same directory as original image as PNG, where
        the suffix '-crop_<i>' is added where i=0 to number of <imgs>.

        <row> is the metadata row for the original image.
        """
        lst_name = x.filename.iloc[0].split(".")

        new_names = []
        for i in range(len(imgs)):
            new_name = ".".join(lst_name[:-1]) + f"-crop_{i}.png"
            Image.fromarray(imgs[i]).convert("L").save(x.path.iloc[0] + "/" + new_name)
            new_names.append(new_name)

        return new_names

    def supplement_label(self, label: str, overwrite=False):
        """Increase diversity of <label> by increasing diversity of resolutions. And
        upsamples label if less than 1000.
            - If True, early exit
            - If False, determine how many images need to be cropped to supplement label.
                1. Check which images can be cropped (i.e. possess at least 2x224 on width or height
                2. Randomly select row for image cropping, and get random crop
                3. Update label table with new image crops, replacing original image.
                4. Save to <label>_upsampled.csv
        """
        # Skip if label already upsampled
        if os.path.isfile(annotations_dir + f"classes/upsampled/{label}.csv") and not overwrite:
            print(f"Upsampling of {label} Already Done!")
            return

        # Get label with assigned images
        df_label = pd.read_csv(annotations_dir + f"classes/{label}.csv")

        # Check if all images exists
        df_label.apply(check_exists, axis=1)

        # Label which images are crops (temporary). Create columns for crop info.
        df_label_up = df_label.assign(crop=False, scaling=1,
                                      x_min=None, x_max=None, y_min=None, y_max=None)

        # How many crops to preferably get per croppable image?     # NOTE: At most num_desired will be 4. At least 1
        if len(df_label) == 1000:
            num_desired = 1
        elif len(df_label) >= 550:      # 550+ images present
            num_desired = 2
        elif len(df_label) >= 400:      # 400+ images present
            num_desired = 3
        else:               # < 400 images present
            num_desired = 4

        remove_from_base = []
        # Loop through all images
        for i in range(len(df_label)):
            # Get metadata for image
            image_idx = df_label.iloc[i].idx
            row = df_label_up[df_label_up.idx == image_idx]

            # Load image, as grayscale
            img = cv2.imread(df_label.iloc[i].path + "/" + df_label.iloc[i].filename, cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.array(Image.open(df_label.iloc[i].path + "/" + df_label.iloc[i].filename))

            # If NoneType or max = 0 or only 0 or 255, remove
            if img is None or img.max() == 0 or np.percentile(img, 0.01) == np.percentile(img, 99.9):
                print("Bad image found!")
                df_label_up = df_label_up[df_label_up.idx != image_idx]
                remove_from_base.append(image_idx)
                continue
            elif len(np.unique(img)) == 2:     # if binary mask, try to recreate
                print("Binary mask found!")
                create_image(df_label.iloc[i])
                img = cv2.imread(df_label.iloc[i].path + "/" + df_label.iloc[i].filename)
                if img is None:
                    img = np.array(Image.open(df_label.iloc[i].path + "/" + df_label.iloc[i].filename))
                # If persists, remove from images
                if len(np.unique(img)) == 2:
                    df_label_up = df_label_up[df_label_up.idx != image_idx]
                    remove_from_base.append(image_idx)
                    continue

            # Get crops if dimensions of image >= 140x140
            if img.shape[0] >= 140 and img.shape[1] >= 140:
                # Normalize
                # if img.max() != 255.0 or img.min() != 0.0:
                #     img = normalize(img) * 255

                # img = Image.fromarray(img).convert("L")

                # Create & save crops
                crop_imgs, crop_info = self.create_crops(img, num_desired)

                # If no images returned (black/white 1/2x image), use original image
                if len(crop_imgs) == 0:
                    continue

                # Else, save new crops
                new_filenames = self.save_crops(crop_imgs, row)

                # Create metadata for new crops
                if len(crop_imgs) > 1:
                    row.at[row.index[0], "x_min"] = crop_info[0]
                    df_crops = row.explode("x_min", ignore_index=True)
                else:
                    df_crops = row
                    df_crops['x_min'] = crop_info[0]
                try:
                    df_crops['x_max'] = crop_info[1]
                except ValueError:
                    print(df_crops)
                    print("X-Maxs", crop_info[1])
                    raise Exception
                df_crops['y_min'] = crop_info[2]
                df_crops['y_max'] = crop_info[3]
                df_crops['scaling'] = crop_info[4]
                df_crops['filename'] = new_filenames
                df_crops['crop'] = True
                # df_crops = df_crops.assign(crop=True, x_max=crop_info[1], y_min=crop_info[2],
                #                            y_max=crop_info[3], scaling=crop_info[4],
                #                            filename=new_filenames)
                # Update metadata dataframe w/ new crops. Replace original image metadata
                df_label_up = pd.concat([df_label_up[df_label_up.idx != image_idx],
                                         df_crops], ignore_index=True)

        # If exceeded 1000, randomly sample back to 1000
        if len(df_label_up) > 1000:
            df_label_up['temp_index'] = list(range(len(df_label_up)))

            to_remove = df_label_up.groupby('idx').sample(n=1)
            if len(to_remove) >= len(df_label_up) - 1000:
                to_remove_temp_index = to_remove.sample(n=len(df_label_up)-1000).temp_index.tolist()
            else:
                # if to_remove is smaller than number of images needed to remove
                to_remove_temp_index = to_remove.temp_index.tolist()
                left_to_remove = (len(df_label_up) - 1000) - len(to_remove)
                # randomly sample other images to remove
                to_remove_temp_index.extend(df_label_up[~df_label_up.temp_index.isin(to_remove.temp_index.tolist())].sample(n=left_to_remove).temp_index.tolist())

            print(f"Overflow! {len(to_remove_temp_index)} removed.")
            df_label_up = df_label_up[~df_label_up.temp_index.isin(to_remove_temp_index)]
            df_label_up = df_label_up.drop(columns='temp_index')

        df_label_up.to_csv(annotations_dir + f"classes/upsampled/{label}.csv", index=False)
        print(f"Successfully Upsampled {label}! {len(df_label)} -> {len(df_label_up)}")

        # Update Base
        if len(remove_from_base) > 0:
            df_label[~df_label.idx.isin(remove_from_base)].to_csv(annotations_dir + f"classes/{label}.csv", index=False)


def construct_label(label, overwrite=True):
    # Directory label
    dir_label = label.replace(" -- ", "-").replace(" ", "_").replace("/", "-").replace(",", "_")
    # Skip if folder already exists
    if os.path.exists(f'/ferrero/cytoimagenet/{dir_label}') and not overwrite:
        return pd.DataFrame()
    elif os.path.exists(f'/ferrero/cytoimagenet/{dir_label}') and overwrite:
        os.system(f'rm -r /ferrero/cytoimagenet/{dir_label}')
        print(f"Removed files for {dir_label}")

    # If upsampled label has less than 287 images, early exit
    df_ = pd.read_csv(f"{annotations_dir}classes/upsampled/{label}.csv")
    if len(df_) < 287:
        return pd.DataFrame()

    # Check exists. Only download those that exist
    exists_series = df_.apply(check_exists, axis=1)
    if not all(exists_series):
        df_ = df_[exists_series]
        df_.to_csv(f"{annotations_dir}classes/upsampled/{label}.csv", index=False)

    # Assign label
    df_["label"] = label.replace(" -- ", "/")

    # Get absolute paths
    full_paths = df_.apply(lambda x: f"{x.path}/{x.filename}", axis=1).tolist()

    # Remove whitespace and conventions
    dir_label = label.replace(" -- ", "-").replace(" ", "_").replace("/", "-").replace(",", "_")
    print(f"Fixed: {label} -> {dir_label}")

    # Verify label directory exists. If not, create directory
    if not os.path.exists(f'/ferrero/cytoimagenet/{dir_label}'):
        os.mkdir(f'/ferrero/cytoimagenet/{dir_label}')

    # Copy images to new directory
    new_filenames = []
    largest_binary_value = bin(len(full_paths))[2:]     # removing '0b' prefix
    for i in range(len(full_paths)):
        new_filename = bin(i)[2:]
        # If not same length of strings, zero extend
        if len(largest_binary_value) != len(new_filename):
            new_filename = '0' * (len(largest_binary_value) - len(new_filename)) + new_filename
        # Add label
        new_filename = f"{dir_label}-{new_filename}." + full_paths[i].split(".")[-1]        # label-00001.(old_extension)
        shutil.copy(full_paths[i], f'/ferrero/cytoimagenet/{dir_label}/{new_filename}')
        new_filenames.append(new_filename)

    # Update new filenames
    df_['filename'] = new_filenames
    # Update new path
    df_["path"] = f'/ferrero/cytoimagenet/{dir_label}'
    print(f"Success! for {label.replace(' -- ', '/')}")
    return df_


def construct_cytoimagenet(labels: list, overwrite: bool = False):
    """Concatenate metadata from <labels> to create cytoimagenet and update
    metadata. Use metadata to copy images into '/ferrero/cytoimagenet/'
        1. Create folder for each label
        2. Copy corresponding images into folders.
        3. Update metadata
        4. Convert all non-PNG images to PNG
    Save metadata in '/ferrero/cytoimagenet/metadata.csv'
    """
    # Get existing metadata if available
    if os.path.exists("/ferrero/cytoimagenet/metadata.csv") and not overwrite:
        df_metadata = pd.read_csv("/ferrero/cytoimagenet/metadata.csv")
    else:
        df_metadata = pd.DataFrame()

    pool = multiprocessing.Pool(20)
    accum_meta = pool.map(construct_label, labels)
    pool.close()
    pool.join()
    # Save new metadata
    df_metadata = pd.concat([df_metadata, pd.concat(accum_meta, ignore_index=True)])
    df_metadata.to_csv("/ferrero/cytoimagenet/metadata.csv", index=False)
    print("Finished Constructing CytoImageNet!")
    print("Checking for non-PNG images...")

    # Check for non-PNG images
    convert_png_cytoimagenet()

    # Check for unreadable images
    cytoimagenet_check_readable(labels)


def convert_png_cytoimagenet():
    """Using CytoImageNet metadata, convert non-PNG images to png in the
    directory.
    """
    df_metadata = pd.read_csv("/ferrero/cytoimagenet/metadata.csv")

    for label in df_metadata.label.unique():
        df_metadata.loc[(df_metadata.label == label)]

        non_png = df_metadata[~df_metadata.filename.str.contains(".png")]

        # Early Exit: If no non-PNG images
        if len(non_png) == 0:
            return

        png_filenames = non_png.apply(to_png, axis=1)

        exists_series = png_filenames.map(lambda x: x is not None)
        # Only update those that were converted successfully
        idx_to_update = non_png[exists_series].idx.tolist()
        idx = df_metadata.idx.isin(idx_to_update)
        df_metadata.loc[idx, "filename"] = png_filenames[exists_series]

        # Print if not exists
        if not all(exists_series):
            print(non_png[~exists_series].label.value_counts())

    # Update metadata
    df_metadata.to_csv("/ferrero/cytoimagenet/metadata.csv", index=False)
    if False:
        df_metadata = pd.read_csv("/ferrero/cytoimagenet/metadata.csv")
        png_in_name = df_metadata.filename.map(lambda x: ".png" in x)
        df_metadata.loc[~png_in_name, 'filename'] = df_metadata.loc[~png_in_name, 'filename'].map(lambda x: ".".join(x.split(".")[:-1]) + ".png")
        if "Unnamed: 0" in df_metadata.columns:
            df_metadata = df_metadata.drop(columns='Unnamed: 0')
        df_metadata.to_csv("/ferrero/cytoimagenet/metadata.csv", index=False)
        print(pd.read_csv("/ferrero/cytoimagenet/metadata.csv"))


def check_normalized(x):
    """Check if image is normalized. If not, normalize and resave image."""
    img = cv2.imread(x.path + "/" + x.filename, cv2.IMREAD_GRAYSCALE)

    if img.min() == 0 and img.max() == 255:
        return True
    else:
        cv2.imwrite(x.path + "/" + x.filename, normalize(img) * 255)


def cytoimagenet_check_readable(labels):
    df_metadata = pd.read_csv("/ferrero/cytoimagenet/metadata.csv")

    # Initialize
    # if 'unreadable' not in df_metadata.columns:
    df_metadata['unreadable'] = False
    df_metadata['checked'] = False

    for label in labels:
        # Check readable
        series_readable = df_metadata[df_metadata.label == label].apply(check_readable, axis=1)
        unreadable_idx = df_metadata[df_metadata.label == label][~series_readable].idx.tolist()

        # Update unreadable images if there are
        if len(unreadable_idx) > 0:
            idx = df_metadata.idx.isin(unreadable_idx)
            df_metadata.loc[idx, 'unreadable'] = True

        # Update labels to be checked
        idx_2 = (df_metadata.label == label)
        df_metadata.loc[idx_2, 'checked'] = True

    # Metadata
    df_metadata.to_csv("/ferrero/cytoimagenet/metadata.csv", index=False)


# CHECK FILE SIZES > 6 KB
def cytoimagenet_check_sizes():
    df = pd.read_csv("/ferrero/cytoimagenet/metadata.csv")

    def file_size_sufficient(x):
        if os.path.getsize(x.path + "/" + x.filename) / 1000 <= 6:
            print(x.dir_name)
            return False
        return True

    sufficient_series = df.apply(file_size_sufficient, axis=1)
    print(len(sufficient_series[~sufficient_series]), " images that are <= 6 KB!")
    print("Affected Datasets: ", df[~sufficient_series].dir_name.unique().tolist())
    print("Num Affected Labels: ", len(df[~sufficient_series].label.unique().tolist()))
    print("Example Affected Labels: ", random.sample(df[~sufficient_series].label.unique().tolist(), 10))
    df[~sufficient_series].to_csv("/ferrero/cytoimagenet/insufficient_filesize.csv", index=False)
    df[sufficient_series].to_csv("/ferrero/cytoimagenet/sufficient_filesize.csv", index=False)


# REMOVING IMAGES <= 6 KB
def cytoimagenet_remove_insufficient():
    """Remove images from cytoimagenet dataset that are less than or equal to
    6 kilobytes in size.

    ==Precondition==:
        - cytoimagenet_check_sizes() was called beforehand.
    """
    df = pd.read_csv("/ferrero/cytoimagenet/insufficient_filesize.csv")

    def remove_from_cytoimagenet(x):
        os.remove(x.path + "/" + x.filename)
    # Remove each image one by one
    df.apply(remove_from_cytoimagenet, axis=1)

    # Check for classes that are <= 287 images. Remove from cytoimagenet completely
    df = pd.read_csv("/ferrero/cytoimagenet/sufficient_filesize.csv")
    count_labels = df.groupby(by=['label']).count().iloc[:, 0]
    below_thresh = count_labels[count_labels < 500].index.tolist()
    for label in below_thresh:
        dir_label = label.replace(" -- ", "-").replace(" ", "_").replace("/", "-").replace(",", "_")
        if os.path.exists(f'/ferrero/cytoimagenet/{dir_label}'):
            os.system(f'rm -r /ferrero/cytoimagenet/{dir_label}')
            print(f"Successfully removed {dir_label}")

    print(len(below_thresh), 'classes removed!')


# REMOVING REDUNDANT IMAGES
def cytoimagenet_remove_duplicates():
    df_metadata = pd.read_csv("/ferrero/cytoimagenet/metadata.csv")
    # Create temporary variable
    df_metadata['duplicate'] = False

    # Get duplicate idx.
    duplicate_idx = df_metadata[df_metadata.duplicated(subset=["idx"])].idx

    for idx in duplicate_idx:
        df_same_idx = df_metadata[df_metadata.idx == idx]
        # Check if same idx is found in more than 1 label
        if df_same_idx.label.nunique() > 1:
            # Choose label with the smallest size to retain images. The rest are duplicates
            labels = df_same_idx.label.unique()
            class_sizes = [len(df_metadata[df_metadata.label == label]) for label in labels]

            kept_label = labels[np.argmin(class_sizes)]
            duplicate_labels = [label for label in labels if label != kept_label]

            # Mark as duplicates
            df_metadata.loc[(df_metadata.idx == idx) & (df_metadata.label.isin(duplicate_labels)), 'duplicate'] = True

    # Remove duplicate images from directory
    if df_metadata.duplicate.sum() > 0:
        # Save unique image metadata
        df_metadata_unique = df_metadata[~df_metadata.duplicate]
        df_metadata_unique.to_csv("/ferrero/cytoimagenet/metadata.csv", index=False)

        # Remove duplicate images
        df_metadata[df_metadata.duplicate].to_csv("/ferrero/cytoimagenet/redundant.csv", index=False)
        df_metadata[df_metadata.duplicate].apply(lambda x: os.remove(x.path + "/" + x.filename), axis=1)
        print(len(df_metadata[df_metadata.duplicate]), " removed!")


# RECREATE METADATA
def cytoimagenet_recreate_metadata(labels):
    """Recreate CytoImageNet metadata from upsampled classes csv files.

    ==Precondition==:
        - all images in cytoimagenet/annotations/classes/upsampled/*.csv are
            used.
    """
    accum_dfs = []

    for label in labels:
        df_ = pd.read_csv(f"{annotations_dir}classes/upsampled/{label}.csv")

        # Fix label
        df_["label"] = label.replace(" -- ", "/")

        # Get absolute paths
        full_paths = df_.apply(lambda x: f"{x.path}/{x.filename}", axis=1).tolist()

        # Remove whitespace and conventions
        dir_label = label.replace(" -- ", "-").replace(" ", "_").replace("/", "-").replace(",", "_")

        # Verify label directory exists. If not, create directory
        if not os.path.exists(f'/ferrero/cytoimagenet/{dir_label}'):
            os.mkdir(f'/ferrero/cytoimagenet/{dir_label}')

        # Copy images to new directory
        new_filenames = []
        largest_binary_value = bin(len(full_paths))[2:]     # removing '0b' prefix
        for i in range(len(full_paths)):
            new_filename = bin(i)[2:]
            # If not same length of strings, zero extend
            if len(largest_binary_value) != len(new_filename):
                new_filename = '0' * (len(largest_binary_value) - len(new_filename)) + new_filename
            # Add label
            new_filename = f"{dir_label}-{new_filename}." + full_paths[i].split(".")[-1]        # label-00001.(old_extension)
            new_filenames.append(new_filename)

        # Update new filenames
        df_['filename'] = new_filenames
        # Update new path
        df_["path"] = f'/ferrero/cytoimagenet/{dir_label}'
        accum_dfs.append(df_)
    df_metadata = pd.concat(accum_dfs, ignore_index=True)
    df_metadata.to_csv("/ferrero/cytoimagenet/metadata.csv", index=False)


def cytoimagenet_fix_metadata():
    """Clean CytoImageNet metadata."""
    df_metadata = pd.read_csv("/ferrero/cytoimagenet/metadata.csv")
    df_metadata.drop(columns=['unreadable', 'checked'], inplace=True)
    df_metadata.microscopy = df_metadata.microscopy.map(lambda x: 'fluorescence' if x == "fluorescence (?)" else x)
    df_metadata.microscopy = df_metadata.microscopy.map(lambda x: 'fluorescence|brightfield|darkfield' if x == "['fluorescence', 'brightfield', 'darkfield']" else x)
    df_metadata.channels = df_metadata.channels.map(lambda x: 'fluorescence|brightfield|darkfield' if x == "['fluorescence', 'brightfield', 'darkfield']" else x)

    df_metadata.to_csv("/ferrero/cytoimagenet/metadata.csv", index=False)


# CONSTRUCT VALIDATION SET
def construct_unused_labels():
    labels = [i.split("classes/")[-1].split(".csv")[0] for i in glob.glob(annotations_dir + "unused_classes/*.csv")]
    pool = multiprocessing.Pool(10)
    accum_meta = pool.map(construct_unused_label, labels, True)
    pool.close()
    pool.join()
    df_val = pd.concat(accum_meta, ignore_index=True)

    # Temp index
    df_val['temp_index'] = list(range(len(df_val)))

    # Convert images to PNG
    non_png = df_val[~df_val.filename.str.contains(".png")]
    if len(non_png) != 0:      # Exit if all PNG
        png_filenames = non_png.apply(to_png, axis=1)
        exists_series = png_filenames.map(lambda x: x is not None)
        # Only update those that were converted successfully
        idx_to_update = non_png[exists_series].temp_index.tolist()
        idx = df_val.temp_index.isin(idx_to_update)
        df_val.loc[idx, "filename"] = png_filenames[exists_series]

    df_val.drop(columns=['temp_index'], inplace=True)
    df_val.to_csv('/ferrero/stan_data/unseen_classes/metadata.csv', index=False)


def construct_unused_label(label, overwrite=True):
    df_ = pd.read_csv(f"{annotations_dir}unused_classes/{label}.csv")

    # Check exists. Only download those that exist
    exists_series = df_.apply(check_exists, axis=1)
    if not all(exists_series):
        df_ = df_[exists_series]
        df_.to_csv(f"{annotations_dir}unused_classes/{label}.csv", index=False)

    # Assign label
    df_["label"] = label.replace(" -- ", "/")

    # Get absolute paths
    full_paths = df_.apply(lambda x: f"{x.path}/{x.filename}", axis=1).tolist()

    # Remove whitespace and conventions
    dir_label = label.replace(" -- ", "-").replace(" ", "_").replace("/", "-").replace(",", "_")
    print(f"Fixed: {label} -> {dir_label}")

    # Verify label directory exists. If not, create directory
    if not os.path.exists('/ferrero/stan_data/unseen_classes/'):
        os.mkdir('/ferrero/stan_data/unseen_classes/')
    if not os.path.exists(f'/ferrero/stan_data/unseen_classes/{dir_label}'):
        os.mkdir(f'/ferrero/stan_data/unseen_classes/{dir_label}')

    # Copy images to new directory
    new_filenames = []
    largest_binary_value = bin(len(full_paths))[2:]     # removing '0b' prefix
    for i in range(len(full_paths)):
        new_filename = bin(i)[2:]
        # If not same length of strings, zero extend
        if len(largest_binary_value) != len(new_filename):
            new_filename = '0' * (len(largest_binary_value) - len(new_filename)) + new_filename
        # Add label
        new_filename = f"{dir_label}-{new_filename}." + full_paths[i].split(".")[-1]        # label-00001.(old_extension)
        shutil.copy(full_paths[i], f'/ferrero/stan_data/unseen_classes/{dir_label}/{new_filename}')
        new_filenames.append(new_filename)

    # Update new filenames
    df_['filename'] = new_filenames
    # Update new path
    df_["path"] = f'/ferrero/stan_data/unseen_classes/{dir_label}'
    print(f"Success! for {label.replace(' -- ', '/')}")
    return df_


# FIX [0, 1] images
def find_improperly_processed_labels():
    df_metadata = pd.read_csv("/ferrero/cytoimagenet/metadata.csv")
    sampled_rows = df_metadata.groupby(by=['dir_name']).sample(n=20, replace=True)

    img_none = []
    img_improcessed = []
    for i in range(len(sampled_rows)):
        x = sampled_rows.iloc[i]
        img = cv2.imread(x.path + "/" + x.filename, cv2.IMREAD_GRAYSCALE)
        if img is None:
            if x.dir_name not in img_none:
                img_none.append(x.dir_name)
        elif img.max() == 1 and x.dir_name not in img_improcessed:
            img_improcessed.append(x.dir_name)
    print(len(img_improcessed), " improperly preprocessed!")
    print(img_improcessed)
    print()
    print(len(img_none), " has None images")
    print(img_none)
    return img_none, img_improcessed


def fix_unreadable_files(label, df_metadata):
    df_ = pd.read_csv(f"{annotations_dir}classes/upsampled/{label}.csv")
    df_.reset_index(drop=True, inplace=True)

    indices_to_recreate = df_metadata[(df_metadata.unreadable) & (df_metadata.label == label)].idx.tolist()

    keep_ilocations = []
    remove_ilocations = []
    for idx in indices_to_recreate:
        x = df_[df_.idx == idx]
        # create_image(x.iloc[0])
        try:
            Image.open(x.path + "/" + x.filename)
            keep_ilocations.append(x.index)
        except:
            remove_ilocations.extend(x.index.tolist())
    # Early exit
    if len(keep_ilocations) == 0 and len(remove_ilocations) == 0:
        return []
    print(keep_ilocations)
    print(remove_ilocations)
    # Assign label
    df_["label"] = label.replace(" -- ", "-")

    # Get absolute paths
    full_paths = df_.apply(lambda x: f"{x.path}/{x.filename}", axis=1).tolist()

    # Remove whitespace and conventions
    dir_label = label.replace(" -- ", "-").replace(" ", "_").replace("/", "-").replace(",", "_")

    # Copy images to new directory
    new_filenames = []
    largest_binary_value = bin(len(full_paths))[2:]     # removing '0b' prefix
    filenames_to_remove = []
    for i in range(len(full_paths)):
        new_filename = bin(i)[2:]
        # If not same length of strings, zero extend
        if len(largest_binary_value) != len(new_filename):
            new_filename = '0' * (len(largest_binary_value) - len(new_filename)) + new_filename
        # Add label
        new_filename = f"{dir_label}-{new_filename}." + full_paths[i].split(".")[-1]        # label-00001.(old_extension)
        if i in keep_ilocations:
            shutil.copy(full_paths[i], f'/ferrero/cytoimagenet/{dir_label}/{new_filename}')
            print("Updated unreadable image for ", label)
        elif i in remove_ilocations:
            print("Removed unreadable image for ", label)
            os.remove(f'/ferrero/cytoimagenet/{dir_label}/{new_filename}')
            filenames_to_remove.append(new_filename)
        new_filenames.append(new_filename)

    # Remove from upsampled metadata
    if len(remove_ilocations) > 0:
        old_filenames_to_remove = []
        for i in remove_ilocations:
            old_filenames_to_remove.append(df_.iloc[i].filename)
        df_[~df_.filename.isin(old_filenames_to_remove)].to_csv(f"{annotations_dir}classes/upsampled/{label}.csv", index=False)

    return filenames_to_remove


def cytoimagenet_find_remove_below_thresh():
    """Find labels in CytoImageNet that are below 287 images. Remove label
    from metadata and dataset."""
    df_metadata = pd.read_csv("/ferrero/cytoimagenet/metadata.csv")
    print('endothelial' not in df_metadata.label.unique())
    df_count = df_metadata.groupby('label').count().iloc[:, 0]

    # Remove classes below 500 images
    labels_to_remove = df_count[df_count < 500].index.tolist()
    print(labels_to_remove)

    for label in labels_to_remove:
        dir_label = label.replace(" -- ", "-").replace(" ", "_").replace("/", "-").replace(",", "_")
        os.system(f"rm -r /ferrero/cytoimagenet/{dir_label}")

    # Save new metadata
    df_metadata[~df_metadata.label.isin(labels_to_remove)].to_csv("/ferrero/cytoimagenet/metadata.csv", index=False)


def cytoimagenet_fix_incorrect_filenaming():
    df_metadata = pd.read_csv("/ferrero/cytoimagenet/metadata.csv")
    old_paths = df_metadata[df_metadata.path.str.contains(",")].path.unique()
    new_paths = [i.replace(",", "_") for i in old_paths]
    for j in range(len(old_paths)):
        os.rename(old_paths[j], new_paths[j])
    df_metadata.path = df_metadata.path.str.replace(",", "_")

    df_metadata.to_csv("/ferrero/cytoimagenet/metadata.csv", index=False)


def main(file):
    pass
    # df = pd.read_csv(file)
    # label = file.replace(annotations_dir + "classes/upsampled/", "").replace(".csv", "")

    # print(f"\n\n{label} currently processing!\n\n\n")

    # Check if images exist. If not, try to create images.
    # exists_series = df.apply(check_exists, axis=1)
    # # If not all exists, filter for only those that exist.
    # if not all(exists_series):
    #     df = df[exists_series]

    # Filter for tensorflow-readable images.
    # readable_series = df.apply(check_readable, axis=1)
    # # If not all are 'readable', try remerging image.
    # if not all(readable_series):
    #     # Recreate Image
    #     df[~readable_series].apply(create_image, axis=1)
    #     # Recheck if images are tensorflow-readable. If not, only filter readable images.
    #     readable_series = df.apply(check_readable, axis=1)
    #     df = df[readable_series]

    # Remove class if less than 287 samples.
    # if len(df) < 287:
    #     print(f"Removed {label}")
    #     os.remove(file)
    #     if os.path.exists(annotations_dir + f"classes/upsampled/{label}.csv"):
    #         os.remove(annotations_dir + f"classes/upsampled/{label}.csv")
    #     if os.path.exists(f"/ferrero/cytoimagenet/{label}"):
    #         shutil.rmtree(f"/ferrero/cytoimagenet/{label}")

    # Check if images are normalized. Normalize in place if not.
    # df.apply(check_normalized, axis=1)

    # Save results
    # df.to_csv(file, index=False)

    # Upsample label
    # Upsampler().supplement_label(label, True)

    # Check if there are RGB images
    # Sample 2 rows from each dataset present
    # name_samples = df[exists_series].groupby(by=["name"]).sample(frac=0.25)
    # name_idx = name_samples.apply(check_grayscale, axis=1)
    # ds_to_grayscale = name_samples[~name_idx].dir_name.tolist()
    # if len(ds_to_grayscale) > 0:
    #     print("List of Datasets with non-RGB images: ", ds_to_grayscale)
    #     for name in ds_to_grayscale:
    #         df[(df.dir_name == name) & (exists_series)].apply(to_grayscale, axis=1)


if __name__ == '__main__' and "D:\\" not in os.getcwd():
    files = glob.glob(annotations_dir + "classes/upsampled/*.csv")

    construct_unused_labels()

    all_labels = [i.split("classes/upsampled/")[-1].split(".csv")[0] for i in glob.glob(annotations_dir + "classes/upsampled/*.csv")]

    df_metadata = pd.read_csv("/ferrero/cytoimagenet/metadata.csv")

    # pool = multiprocessing.Pool(30)
    # pool.map(main, files)
    # pool.close()
    # pool.join()

    # construct_cytoimagenet(all_labels, True)

    # print("Metadata Length: ", len(pd.read_csv("/ferrero/cytoimagenet/metadata.csv")))
    #
    # cytoimagenet_recreate_metadata(all_labels)

    if False:
        df_metadata = pd.read_csv("/ferrero/cytoimagenet/metadata.csv")
        x = df_metadata[df_metadata.unreadable]
        unreadable_labels = (x.label.unique().tolist())
        print(unreadable_labels)
        to_remove = []
        for label in unreadable_labels:
            to_remove.extend(fix_unreadable_files(label, df_metadata))
        print(to_remove)
        if len(to_remove) > 0:
            df_metadata[~df_metadata.filename.isin(to_remove)].to_csv("/ferrero/cytoimagenet/metadata.csv", index=False)

    # Remove labels below 500 images
    # find_remove_below_thresh()

    # Fix filenaming
    cytoimagenet_fix_incorrect_filenaming()

    # Check filesizes
    # cytoimagenet_check_sizes()
