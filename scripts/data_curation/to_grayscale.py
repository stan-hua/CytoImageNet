"""Convert RGB images to Grayscale.
Assumption:
    - channels do not represent different stains

"""
import numpy as np
import os
from PIL import Image

# javabridge.start_vm(class_path=bf.JARS)

if "D:\\" not in os.getcwd():
    annotations_dir = "/home/stan/cytoimagenet/annotations/"
    data_dir = '/ferrero/stan_data/'
else:
    annotations_dir = "/annotations/"
    data_dir = 'M:/ferrero/stan_data/'

# Indiscriminately convert all RGB images to grayscale. Save in its own dir
