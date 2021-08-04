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

# Only use CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


# PATHS
if "D:\\"  in os.getcwd():
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
    try:
        a = cv2.imread(x.path + "/" + x.filename, cv2.IMREAD_GRAYSCALE)
        if a is None:
            return False
        else:
            return True
    except:
        try:
            print(f"Creating images for {x.dir_name}")
            start = time.perf_counter()
            create_image(x)
            print(f"Image Created in {time.perf_counter()-start} seconds!")
            if os.path.exists(x.path + "/" + x.filename):
                return True
            return False
        except:
            return False

    # if os.path.exists(x.path + "/" + x.filename):
    #     return True

    # If file does not exist
    # try:
    #     print(f"Creating images for {x.dir_name}")
    #     start = time.perf_counter()
    #     create_image(x)
    #     print(f"Image Created in {time.perf_counter()-start} seconds!")
    #     if os.path.exists(x.path + "/" + x.filename):
    #         return True
    #     return False
    # except:
    #     return False


def check_file_extension(x):
    """Check if image as PNG (or JPEG, BMP) exists. If not, create PNG copy.
    """
    # Filename with PNG extension
    new_name = ".".join(x.filename.split(".")[:-1]) + ".png"

    # Check if file is of supported image format
    readable_ext = any([ext in x.filename.lower() for ext in ["png", "jpeg", "jpg", "bmp"]])

    if not readable_ext and not os.path.exists(x.path + "/" + new_name):
        img = Image.open(x.path + "/" + x.filename)
        img.convert("L").save(x.path + "/" + new_name)
        print(f"Creating PNG copy for {x.dir_name}!")


def rename_extension(x):
    """Given filename, rename file extension to ".png" if not in supported
    image format (PNG, JPEG, BMP).
    """
    # Check if file is of supported image format
    readable_ext = any([ext in x.lower() for ext in ["png", "jpeg", "jpg", "bmp"]])

    if not readable_ext:
        return ".".join(x.split(".")[:-1]) + ".png"
    return x


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


# TODO: Check if images are problematic
def check_problematic(x):
    """Check if each image is not completely black/white by a set threshold."""
    img = cv2.imread(x.path + "/" + x.filename, cv2.IMREAD_GRAYSCALE)

    if img.min() == img.max():
        print("Completely Black/White Image!")
        return True
    # if sum([img.min(), img.max()]) <= 480 and sum([img.min(), img.max()]) >= 30:
    #     print("Lacking in variation!")
    #     return True
    return False


def check_constant(x):
    try:
        img = cv2.imread(x.path + "/" + x.filename, cv2.IMREAD_GRAYSCALE)
    except:
        return True
    if img.min() == img.max():
        print("Found Constant Image!")
        return True
    return False


def check_readable(x):
    """Check if Image is Readable by Tensorflow.
    """
    image = tf.io.read_file(x.path + "/" + x.filename)
    image = tf.image.decode_png(image)
    # image = tf.image.convert_image_dtype(image, tf.float32)
    if (tf.reduce_max(image) == tf.reduce_min(image)) or image is None:         # Check if image is constant, or image is None
        return False
    return True


def get_slicers(height, width, num_crops, min_height, min_width):
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
        # print("Number of Quadrants sampled from: ", len(quadrant_indices))

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


def create_crops(img: Image.Image, num_crops: int):
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
    (x_mins, x_maxs, y_mins, y_maxs), scaling = get_slicers(
        img.size[0], img.size[1],
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
        img_crop = img.crop((x_mins[i], y_mins[i], x_maxs[i], y_maxs[i]))
        # Save crop if not white/black
        if sum(img_crop.getextrema()) <= 480 and sum(img_crop.getextrema()) >= 15:
            img_crops.append(img_crop)
            used_xmins.append(x_mins[i])
            used_xmaxs.append(x_maxs[i])
            used_ymins.append(y_mins[i])
            used_ymaxs.append(y_maxs[i])
            used_scaling.append(scaling[i])

    return img_crops, (used_xmins, used_xmaxs, used_ymins, used_ymaxs, used_scaling)


def save_crops(imgs: list, x) -> list:
    """Return list of new filenames for crops.

    Save image crops <imgs> in the same directory as original image as PNG, where
    the suffix '-crop_<i>' is added where i=0 to number of <imgs>.

    <row> is the metadata row for the original image.
    """
    lst_name = x.filename.iloc[0].split(".")

    new_names = []
    for i in range(len(imgs)):
        new_name = ".".join(lst_name[:-1]) + f"-crop_{i}.png"
        imgs[i].convert("L").save(x.path.iloc[0] + "/" + new_name)
        new_names.append(new_name)

    return new_names


def supplement_label(label: str):
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
    if os.path.isfile(annotations_dir + f"classes/upsampled/{label}.csv"):
        print(f"Upsampling of {label} Already Done!")
        return

    # Get label with assigned images
    df_label = pd.read_csv(annotations_dir + f"classes/{label}.csv")

    # Check if label has >= 1000 images
    n = len(df_label) - 1000

    # Label which images are crops (temporary). Create columns for crop info.
    df_label_up = df_label.assign(crop=False, scaling=1,
                                  x_min=None, x_max=None, y_min=None, y_max=None)

    # How many crops to preferably get per croppable image?     # NOTE: At most num_desired will be 4. At least 1
    num_desired = max(min(round(abs(n) / len(df_label)), 4), 1)

    # Loop through all images
    for i in range(len(df_label)):
        # Get metadata for image
        image_idx = df_label.iloc[i].idx
        row = df_label_up[df_label_up.idx == image_idx]

        # Load image, as grayscale
        img = cv2.imread(df_label.iloc[i].path + "/" + df_label.iloc[i].filename)

        # Get crops if dimensions of image >= 140x140
        if img.shape[0] >= 140 and img.shape[1] >= 140:
            min_max = img.min(), img.max()
            # Mark black/white images
            if min_max[0] == min_max[1]:
                print("ERROR! Black/White Image!")
                print(label)
                print(image_idx)
                print(row.path.iloc[0] + "/" + row.filename.iloc[0])
                # Skip
                continue
            # Mark problematic images
            elif sum(min_max) > 480 or sum(min_max) < 30:
                print(label)
                print(image_idx)
                print(row.path.iloc[0] + "/" + row.filename.iloc[0])
            # Normalize
            if img.max() != 255.0 or img.min() != 0.0:
                img = normalize(img)
                img = img * 255

            img = Image.fromarray(img).convert("L")

            # Create & save crops
            crop_imgs, crop_info = create_crops(img, num_desired)

            # If no images returned (black/white 1/2x image), continue
            if len(crop_imgs) == 0:
                continue

            # Else, save new crops
            new_filenames = save_crops(crop_imgs, row)

            # Create metadata for new crops
            row.at[row.index[0], "x_min"] = crop_info[0]
            df_crops = row.explode("x_min", ignore_index=True)
            df_crops = df_crops.assign(crop=True, x_max=crop_info[1], y_min=crop_info[2],
                                       y_max=crop_info[3], scaling=crop_info[4],
                                       filename=new_filenames)

            # Update metadata dataframe w/ new crops. Replace original image metadata
            df_label_up = pd.concat([df_label_up[df_label_up.idx != image_idx],
                                     df_crops], ignore_index=True)

    # If exceeded 1000, randomly sample back to 1000
    if len(df_label_up) > 1000:
        df_label_up = df_label_up.sample(n=1000)

    df_label_up.to_csv(annotations_dir + f"classes/upsampled/{label}.csv", index=False)
    print(f"Successfully Upsampled {label}!")


def list_invalids():
    """Get invalid files from IDR"""
    df = pd.read_csv(annotations_dir + "invalid/invalid_images.csv")

    df = df[df.database == 'Image Data Resource']

    df_idr = pd.DataFrame()
    for name in df.name.unique():
        df_spec = df[df.name == name]
        abb = df_spec.dir_name.iloc[0]
        paths = []
        filenames = []
        for i in range(len(df_spec)):
            try:
                refs = df_spec.iloc[i:i+1].apply(get_file_references, axis=1).iloc[0]
                paths.extend(refs[0])
                filenames.extend(refs[1])
            except:
                paths.append(df_spec.iloc[i].path)
                filenames.append(df_spec.iloc[i].filename)

        df_idr = pd.concat([df_idr,
                            pd.DataFrame({"dataset": [name] * len(paths), "abb": [abb] * len(paths),
                                          "path": paths, "filename": filenames})])

    df_idr.path = df_idr.path.map(lambda x: "/".join(x.split("/")[4:]))
    df_idr.to_csv(annotations_dir + "invalid/invalid_idr_images.csv", index=False)


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

    # Accumulator
    accum_meta = []

    for label in labels:
        # Skip if folder already exists
        if os.path.exists(f'/ferrero/cytoimagenet/{label}') and not overwrite:
            continue
        elif os.path.exists(f'/ferrero/cytoimagenet/{label}') and overwrite:
            # Delete Existing Files
            shutil.rmtree(f'/ferrero/cytoimagenet/{label}', ignore_errors=False, onerror=None)

        df_ = pd.read_csv(f"{annotations_dir}classes/upsampled/{label}.csv")
        # Assign label
        df_["label"] = label

        # Only get non-constant tensorflow-readable images
        readable_series = df_.apply(check_readable, axis=1)
        df_ = df_[readable_series]

        # Get absolute paths
        full_paths = df_.apply(lambda x: f"{x.path}/{x.filename}", axis=1).tolist()
        # Verify label directory exists. If not, create directory
        if not os.path.exists(f'/ferrero/cytoimagenet/{label}'):
            os.mkdir(f'/ferrero/cytoimagenet/{label}')

        # Copy images to new directory
        for i in range(len(full_paths)):
            shutil.copy(full_paths[i], f'/ferrero/cytoimagenet/{label}')

        # Update new path
        df_["path"] = f'/ferrero/cytoimagenet/{label}'
        accum_meta.append(df_)

    if len(accum_meta) > 0:
        # Save new metadata
        df_metadata = pd.concat([df_metadata, pd.concat(accum_meta)])
        df_metadata.to_csv("/ferrero/cytoimagenet/metadata.csv", index=False)

    # Check for non-PNG images
    convert_png_cytoimagenet()


def convert_png_cytoimagenet():
    """Using CytoImageNet metadata, convert non-PNG images to png in the
    directory.
    """
    df_metadata = pd.read_csv("/ferrero/cytoimagenet/metadata.csv")
    if len(df_metadata) == 0:
        return

    non_png = df_metadata[~df_metadata.filename.str.contains(".png")]

    # Early Exit: If no non-PNG images
    if len(non_png) == 0:
        return

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


def check_normalized(x):
    """Check if image is normalized. If not, normalize and resave image."""
    img = cv2.imread(x.path + "/" + x.filename, cv2.IMREAD_GRAYSCALE)

    if img.min() == 0 and img.max() == 255:
        return True
    else:
        cv2.imwrite(x.path + "/" + x.filename, normalize(img))


def cytoimagenet_check():
    """Check directory for the following:
        1. Problematic Images
            - constant
            -
        2. Non-PNG images
    """
    df_metadata = pd.read_csv("/ferrero/cytoimagenet/metadata.csv")
    problems_series = df_metadata.apply(check_problematic, axis=1)

    if any(problems_series):
        df_metadata[problems_series].apply(lambda x: print(x.label + "\n\t" + x.filename), axis=1)


def cytoimagenet_check_readable(label):
    df_metadata = pd.read_csv("/ferrero/cytoimagenet/metadata.csv")

    # Initialize
    if 'unreadable' not in df_metadata.columns:
        df_metadata['unreadable'] = False
        df_metadata['checked'] = False

    # Check if label was checked. If so, skip
    if all(df_metadata[df_metadata.label == label].checked):
        print(f"{label} already checked!")
        return

    # Check readable
    series_readable = df_metadata[df_metadata.label == label].apply(check_readable, axis=1)
    unreadable_idx = df_metadata[df_metadata.label == label][~series_readable].idx.tolist()

    print(unreadable_idx)

    # Update unreadable images if there are
    if len(unreadable_idx) > 0:
        idx = df_metadata.idx.isin(unreadable_idx)
        df_metadata.loc[idx, 'unreadable'] = True

    # Update labels to be checked
    idx_2 = (df_metadata.label == label)
    df_metadata.loc[idx_2, 'checked'] = True

    # Metadata
    df_metadata.to_csv("/ferrero/cytoimagenet/metadata.csv", index=False)


def main(file):
    df = pd.read_csv(file)
    label = file.replace(annotations_dir + "classes/", "").replace(".csv", "")

    print(f"\n\n{label} currently processing!\n\n\n")

    # Check if images exist. If not, try to create images.
    exists_series = df.apply(check_exists, axis=1)
    # If not all exists, filter for only those that exist.
    if not all(exists_series):
        df = df[exists_series]

    # Filter for tensorflow-readable images.
    readable_series = df.apply(check_readable, axis=1)
    # If not all are 'readable', try remerging image.
    if not all(readable_series):
        # Recreate Image
        df[~readable_series].apply(create_image, axis=1)
        # Recheck if images are tensorflow-readable. If not, only filter readable images.
        readable_series = df.apply(check_readable, axis=1)
        df = df[readable_series]

    # Remove class if less than 287 samples.
    if len(df) < 287:
        print(f"Removed {label}")
        os.remove(file)
        if os.path.exists(annotations_dir + f"classes/upsampled/{label}.csv"):
            os.remove(annotations_dir + f"classes/upsampled/{label}.csv")
        if os.path.exists(f"/ferrero/cytoimagenet/{label}"):
            shutil.rmtree(f"/ferrero/cytoimagenet/{label}")

    # Check if images are normalized. Normalize if not.
    df.apply(check_normalized, axis=1)

    # Save results
    df.to_csv(file, index=False)

    # Upsample label
    supplement_label(label)

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
    files = glob.glob(annotations_dir + "classes/*.csv")

#     chosen = ['human', 'nucleus', 'cell membrane',
#               'white blood cell', 'kinase',
#               'wildtype', 'difficult',
#               'nematode', 'yeast', 'bacteria',
#               'vamp5 targeted', 'uv inactivated sars-cov-2',
#               'trophozoite', 'tamoxifen', 'tankyrase inhibitor',
#               'dmso', 'rho associated kinase inhibitor', 'rna',
#               'rna synthesis inhibitor', 'cell body'
#               ]
#
    all_labels = [i.split("classes/")[-1].split(".csv")[0] for i in glob.glob(annotations_dir + "classes/*.csv")]

    import multiprocessing
    a_pool = multiprocessing.Pool(25)
    a_pool.map(main, files)

    print("Constructing CytoImageNet!")
    construct_cytoimagenet(all_labels, True)
#
#     for label in all_labels:
#         cytoimagenet_check_readable(label)

    # df_metadata = pd.read_csv("/ferrero/cytoimagenet/metadata.csv")
    # df_unread = df_metadata[(df_metadata.unreadable) & (df_metadata.label != "fibronectin")]
    #
    # print(df_unread.label.value_counts())
    #
    # for i in range(len(df_unread)):
    #     x = df_unread.iloc[i]
    #     img = cv2.imread(x.path + "/" + x.filename)
    #     tf_img = tf.io.read_file(x.path + "/" + x.filename)
    #     tf_img = tf.image.decode_png(tf_img)
    #     # Check if max == 0
    #     if img.max() == 0 or tf.reduce_max(tf_img) == 0:
    #         print(f"Max == 0 for image idx {x.idx} with label {x.label}")
    #
    #     # Try to remerge original image
    #     df_label = pd.read_csv(f"/home/stan/cytoimagenet/annotations/classes/{x.label}.csv")
    #     old_x = df_label[df_label.idx == x.idx]
    #
    #     # If record not found
    #     if len(old_x) == 0:
    #         print(f"Image idx {x.idx} not present in records!")
    #         continue
    #
    #     old_x = old_x.iloc[0]
    #     create_image(old_x)
    #
    #     remade_img = cv2.imread(old_x.path + "/" + old_x.filename)
    #     if remade_img is None:
    #         print(f"Remerging for Image Idx {x.idx} of class {x.label} produces None. Consider deleting.")
    #         continue
    #     # If not None
    #     if remade_img.max() > 0:
    #         print("Successfully recreated image with Max > 0")
    #     else:
    #         print("Failed to recreate image with Max > 0! Consider rechecking merging process, or delete.")
