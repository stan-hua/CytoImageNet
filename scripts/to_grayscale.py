"""Convert RGB images to Grayscale.
Assumption:
    - channels represent different stains

"""

from clean_metadata import get_data_paths
import javabridge
import bioformats as bf

import numpy as np
import os
from PIL import Image

javabridge.start_vm(class_path=bf.JARS)

if "D:\\" not in os.getcwd():
    annotations_dir = "/home/stan/cytoimagenet/annotations/"
    data_dir = '/ferrero/stan_data/'
else:
    annotations_dir = "D:/projects/cytoimagenet/annotations/"
    data_dir = 'M:/ferrero/stan_data/'

dir_name = "idr0067"

possible_files = get_data_paths(dir_name)

# Are each channel individual grayscale images (e.g. different stains)
channels_meaningful = True

# Indiscriminately convert all RGB images to grayscale. Save in dir
for i in range(len(possible_files[1])):
    if ".png" in possible_files[1][i]:  # skip if already converted
        continue

    img = bf.load_image(possible_files[0][i] + "/" + possible_files[1][i])

    if len(img.shape) == 3:
        if channels_meaningful:
            # Normalize with respect to each channel
            for n in range(img.shape[2]):
                img[:, :, n] = img[:, :, n] / img[:, :, n].max()

        # Average along channels
        img = img.mean(axis=2)
    elif len(img.shape) == 2:
        img = img / img.max()
    else:
        raise NotImplementedError(f"Case Not Handled when Channels == {len(img.shape)}!")
    # Normalize back to 0-255
    img = img * 255

    Image.fromarray(img).convert("L").save(f"{data_dir}{dir_name}/{possible_files[1][i].replace('.dv', '.png')}")

print("Successful!")

javabridge.kill_vm()
