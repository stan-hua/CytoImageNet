import os
import pandas as pd

if "D:\\" not in os.getcwd():
    annotations_dir = "/home/stan/cytoimagenet/annotations/"
    data_dir = '/ferrero/stan_data/'
else:
    annotations_dir = "M:/home/stan/cytoimagenet/annotations/"
    data_dir = 'M:/ferrero/stan_data/'

dir_name = "bbbc021"

df_labels_1 = pd.read_csv(f"{data_dir}{dir_name}/BBBC021_v1_image.csv")
df_labels_2 = pd.read_csv(f"{data_dir}{dir_name}/BBBC021_v1_moa.csv")
df_labels = pd.merge(df_labels_1, df_labels_2, how="left",
                     left_on=["Image_Metadata_Compound",
                              "Image_Metadata_Concentration"],
                     right_on=["compound", "concentration"])
df_labels = df_labels[~df_labels.moa.isna()].reset_index(drop=True)
