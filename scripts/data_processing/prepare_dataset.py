from preprocessor import create_image, normalize
from scripts.data_curation.analyze_metadata import get_df_counts

import glob
import multiprocessing
import os
import random
import shutil
import sys

import PIL
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# Only use CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# PATHS
if "D:\\" in os.getcwd():
    annotations_dir = "M:/home/stan/cytoimagenet/annotations/"
    scripts_dir = 'M:/home/stan/cytoimagenet/scripts'
    plot_dir = "M:/home/stan/cytoimagenet/figures/classes/"
else:
    annotations_dir = "/home/stan/cytoimagenet/annotations/"
    scripts_dir = '/home/stan/cytoimagenet/scripts'
    plot_dir = "/home/stan/cytoimagenet/figures/classes/"

# Import modules from other scripts
# sys.path.append(f"{scripts_dir}/data_processing")
# sys.path.append(f"{scripts_dir}/data_curation")





class HelperFunctions:
    @staticmethod
    def check_exists(x):
        """Return True if image exists at <x>.

        If False, use dataset-specific method to create image. Raise Exception if
        image creation failed.
        """
        # Attempt to open using Open-CV -> PIL.
        img = cv2.imread(x.path + "/" + x.filename, cv2.IMREAD_GRAYSCALE)
        if img is None:
            try:
                img = np.array(PIL.Image.open(x.path + "/" + x.filename))
            except:
                img = None

        if img is None or img.max() == 1:
            create_image(x)
            if os.path.exists(x.path + "/" + x.filename):
                return True
            return False
        return True

    @staticmethod
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

    @staticmethod
    def check_readable(x):
        """Check if Image is Readable by Tensorflow.
        """
        try:
            image = PIL.Image.open(x.path + "/" + x.filename)
            return True
        except PIL.UnidentifiedImageError:
            print("PIL.UnidentifiedImageError!")
            return False

    @staticmethod
    def check_normalized(x):
        """Check if image is normalized. If not, normalize and resave image."""
        img = cv2.imread(x.path + "/" + x.filename, cv2.IMREAD_GRAYSCALE)

        if img.min() == 0 and img.max() == 255:
            return True
        else:
            cv2.imwrite(x.path + "/" + x.filename, normalize(img) * 255)

    @staticmethod
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

    @staticmethod
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


class Upsampler:
    """Upsampler Class. Upsamples class by taking a crop of varying resolution
    in each window of a 2x2 grid."""

    def get_slicers(self, height, width, num_crops, min_height, min_width):
        """Return <num_crops> number of image slicers to crop image into <num_crops>
        based on exponentially decaying resolution. In the form (x_mins, x_maxs,
        y_mins, y_maxs), where each element is a parallel list with the others.


        NOTE: num_crops is at most 4.
        NOTE: x refers to width (or columns). y refers to height (or rows).

        Example of image slicing:
            img[y_offset+y_min: y_offset+y_max, x_offset+x_min: x_offset+x_max]
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
        for height_offset in [0, np.floor(height / 2)]:
            for width_offset in [0, np.floor(width / 2)]:
                # Store y, x coordinates
                quadrant_indices.append((height_offset, width_offset))
        quadrant_indices = np.array(quadrant_indices)
        if len(quadrant_indices) > num_crops:
            quadrant_indices = quadrant_indices[np.random.choice(len(quadrant_indices), num_crops, replace=False)]

        n = 0
        x_mins = []
        x_maxs = []
        y_mins = []
        y_maxs = []
        for height_offset, width_offset in quadrant_indices:
            curr_width = np.floor(width * resolutions[n])
            curr_height = np.floor(height * resolutions[n])

            # If current resolutions are 1/2, don't change anchor point (x_min, x_max)
            if resolutions[n] == 1/2:
                x_min = width_offset
                y_min = height_offset
                x_max = width_offset + curr_width
                y_max = height_offset + curr_height
            # Else, randomly choose anchor point in window
            else:
                x_interval = np.floor(width / 2) - curr_width
                y_interval = np.floor(height / 2) - curr_height

                x_min = width_offset + random.randint(0, x_interval)
                x_max = x_min + curr_width

                y_min = height_offset + random.randint(0, y_interval)
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
            # Save crop if mean pixel intensity is greater than 1 and less
            # than 254. And 75th percentile is not 0.
            if 1 < img_crop.mean() < 254 and np.percentile(img_crop, 75) != 0:
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
        df_label.apply(HelperFunctions.check_exists, axis=1)

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


class CytoImageNetCreation:
    # Helper Function for construct_cytoimagenet
    @staticmethod
    def construct_label(label, overwrite=True):
        """For potential class <label>, make a copy. Rename the file according to
        its label and a binary number with each label. Store in CytoImageNet
        directory. Return dataframe of class metadata with updated filename and
        path.
        """
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
        exists_series = df_.apply(HelperFunctions.check_exists, axis=1)
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

    # Construct CytoImageNet directory
    @staticmethod
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
        try:
            accum_meta = pool.map(CytoImageNetCreation.construct_label, labels)
            pool.close()
            pool.join()
        except Exception as e:
            pool.close()
            pool.join()
            print(e)
            raise Exception("Error occured!")
        # Save new metadata
        df_metadata = pd.concat([df_metadata, pd.concat(accum_meta, ignore_index=True)])
        df_metadata.to_csv("/ferrero/cytoimagenet/metadata.csv", index=False)
        print("Finished Constructing CytoImageNet!")

        # Check for non-PNG images
        CytoImageNetCreation.fix_non_png()

        # Check for unreadable images
        CytoImageNetCreation.cytoimagenet_check_readable(labels)

        # Remove duplicates
        CytoImageNetCreation.cytoimagenet_remove_duplicates()

        # Remove classes below threshold
        CytoImageNetCreation.cytoimagenet_remove_classes_below_thresh()

        # Fix illegal directory filenames
        CytoImageNetCreation.cytoimagenet_fix_incorrect_filenaming()

        # Update metadata
        CytoImageNetCreation.cytoimagenet_add_category()

    # ==Metadata-Specific Functions==
    @staticmethod
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

        CytoImageNetCreation.cytoimagenet_add_category()

    @staticmethod
    def cytoimagenet_fix_metadata():
        """Clean CytoImageNet metadata."""
        df_metadata = pd.read_csv("/ferrero/cytoimagenet/metadata.csv")
        df_metadata.drop(columns=['unreadable', 'checked'], inplace=True)
        df_metadata.microscopy = df_metadata.microscopy.map(lambda x: 'fluorescence' if x == "fluorescence (?)" else x)
        df_metadata.microscopy = df_metadata.microscopy.map(lambda x: 'fluorescence|brightfield|darkfield' if x == "['fluorescence', 'brightfield', 'darkfield']" else x)
        df_metadata.channels = df_metadata.channels.map(lambda x: 'fluorescence|brightfield|darkfield' if x == "['fluorescence', 'brightfield', 'darkfield']" else x)

        df_metadata.to_csv("/ferrero/cytoimagenet/metadata.csv", index=False)

    # ==Dataset Quality Assessment==
    @staticmethod
    def fix_non_png():
        """Using CytoImageNet metadata, convert non-PNG images to png in the
        directory.
        """
        df_metadata = pd.read_csv("/ferrero/cytoimagenet/metadata.csv")
        non_png = df_metadata[~df_metadata.filename.str.contains(".png")]

        # Early Exit: If no non-PNG images
        if len(non_png) == 0:
            return

        png_filenames = non_png.apply(HelperFunctions.to_png, axis=1)

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

    @staticmethod
    def fix_unreadable(label, df_metadata):
        """Look for unreadable images. Recreate or remove."""
        df_ = pd.read_csv(f"{annotations_dir}classes/upsampled/{label}.csv")
        df_.reset_index(drop=True, inplace=True)

        indices_to_recreate = df_metadata[(df_metadata.unreadable) & (df_metadata.label == label)].idx.tolist()

        keep_ilocations = []
        remove_ilocations = []
        for idx in indices_to_recreate:
            x = df_[df_.idx == idx]
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

        # Copy images to new directory or remove from CytoImageNet directory
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

    @staticmethod
    def cytoimagenet_check_readable(labels):
        df_metadata = pd.read_csv("/ferrero/cytoimagenet/metadata.csv")

        # Initialize
        # if 'unreadable' not in df_metadata.columns:
        df_metadata['unreadable'] = False
        df_metadata['checked'] = False

        for label in labels:
            # Check readable
            series_readable = df_metadata[df_metadata.label == label].apply(HelperFunctions.check_readable, axis=1)
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

    @staticmethod
    def find_improperly_processed_labels():
        """Look for images that were saved as [0, 1] instead of [0, 255]."""
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

    @staticmethod
    def cytoimagenet_remove_classes_below_thresh():
        """Find labels in CytoImageNet that are below 287 images. Remove labels
        from dataset."""
        df_metadata = pd.read_csv("/ferrero/cytoimagenet/metadata.csv")
        df_count = df_metadata.groupby('label').count().iloc[:, 0]

        # Remove classes below 500 images
        labels_to_remove = df_count[df_count < 500].index.tolist()

        for label in labels_to_remove:
            dir_label = label.replace(" -- ", "-").replace(" ", "_").replace("/", "-").replace(",", "_")
            os.system(f"rm -r /ferrero/cytoimagenet/{dir_label}")

        # Save new metadata
        df_metadata[~df_metadata.label.isin(labels_to_remove)].to_csv("/ferrero/cytoimagenet/metadata.csv", index=False)

    @staticmethod
    def cytoimagenet_fix_incorrect_filenaming():
        """Renaming files with illegal characters in name."""
        df_metadata = pd.read_csv("/ferrero/cytoimagenet/metadata.csv")
        old_paths = df_metadata[df_metadata.path.str.contains(",")].path.unique()
        new_paths = [i.replace(",", "_") for i in old_paths]
        for j in range(len(old_paths)):
            os.rename(old_paths[j], new_paths[j])
        df_metadata.path = df_metadata.path.str.replace(",", "_")

        df_metadata.to_csv("/ferrero/cytoimagenet/metadata.csv", index=False)

    @staticmethod
    def cytoimagenet_remove_duplicates():
        """Looks for duplicate image idx across labels. Removes duplicates."""
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

    @staticmethod
    def cytoimagenet_add_category():
        """Update metadata to include label to category mapping."""
        df_metadata = pd.read_csv("/ferrero/cytoimagenet/metadata.csv")
        df_counts = get_df_counts()

        # Hold updated labels that replaces '/' to '-'
        modified_labels = df_metadata.label.map(lambda x: x.replace("/", "-"))
        mapping_label = dict(zip(df_counts.label, df_counts.category))
        df_metadata['category'] = modified_labels.map(lambda x: mapping_label[x])
        df_metadata.to_csv("/ferrero/cytoimagenet/metadata.csv", index=False)


def main(file,
         verify_exists=False,
         verify_readable=False,
         verify_class_size=False,
         verify_normalized=False,
         verify_grayscale=False
         ):
    df = pd.read_csv(file)
    label = file.replace(annotations_dir + "classes/upsampled/", "").replace(".csv", "")

    print(f"\n\n{label} currently processing!\n\n\n")

    # Check if images exist. If not, try to create images.
    if verify_exists:
        exists_series = df.apply(HelperFunctions.check_exists, axis=1)
        # If not all exists, filter for only those that exist.
        if not all(exists_series):
            df = df[exists_series]

    if verify_readable:
        # Filter for tensorflow-readable images.
        readable_series = df.apply(HelperFunctions.check_readable, axis=1)
        # If not all are 'readable', try remerging image.
        if not all(readable_series):
            # Recreate Image
            df[~readable_series].apply(create_image, axis=1)
            # Recheck if images are tensorflow-readable. If not, only filter readable images.
            readable_series = df.apply(HelperFunctions.check_readable, axis=1)
            df = df[readable_series]

    if verify_class_size:
        # Remove class if less than 287 samples.
        if len(df) < 287:
            print(f"Removed {label}")
            os.remove(file)
            if os.path.exists(annotations_dir + f"classes/upsampled/{label}.csv"):
                os.remove(annotations_dir + f"classes/upsampled/{label}.csv")
            if os.path.exists(f"/ferrero/cytoimagenet/{label}"):
                shutil.rmtree(f"/ferrero/cytoimagenet/{label}")

    if verify_normalized:
        # Check if images are normalized. Normalize in place if not.
        df.apply(HelperFunctions.check_normalized, axis=1)

        # Save results
        df.to_csv(file, index=False)

    if verify_grayscale:
        name_samples = df[exists_series].groupby(by=["name"]).sample(frac=0.25)
        name_idx = name_samples.apply(HelperFunctions.check_grayscale, axis=1)
        ds_to_grayscale = name_samples[~name_idx].dir_name.tolist()
        if len(ds_to_grayscale) > 0:
            print("List of Datasets with non-RGB images: ", ds_to_grayscale)
            for name in ds_to_grayscale:
                df[(df.dir_name == name) & (exists_series)].apply(HelperFunctions.to_grayscale, axis=1)

    # Upsample label
    Upsampler().supplement_label(label, True)


if __name__ == '__main__' and "D:\\" not in os.getcwd():
    redo_upsampling = False
    reconstruct_cytoimagenet = False

    # Get label-specific metadata files
    files = glob.glob(annotations_dir + "classes/upsampled/*.csv")
    all_labels = [i.split("classes/upsampled/")[-1].split(".csv")[0] for i in glob.glob(annotations_dir + "classes/upsampled/*.csv")]

    # Upsample classes
    if redo_upsampling:
        pool = multiprocessing.Pool(30)
        try:
            pool.map(main, files)
        except Exception as e:
            pool.close()
            pool.join()

    # Construct CytoImageNet
    if reconstruct_cytoimagenet:
        CytoImageNetCreation.construct_cytoimagenet(all_labels, True)
        print("Metadata Length: ", len(pd.read_csv("/ferrero/cytoimagenet/metadata.csv")))

    # Update metadata
    CytoImageNetCreation.cytoimagenet_add_category()


