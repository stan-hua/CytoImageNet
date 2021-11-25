from describe_dataset import str_to_eval
from scripts.data_curation.download_data import download
import pandas as pd
import os

home_dir = "/home/stan"
df = pd.read_csv(f"{home_dir}/cytoimagenet/annotations/datasets_info.csv")
df = df.applymap(str_to_eval)

eligible = df[df.dir_name.str.contains("idr")]

for index in eligible.index:
    download_files = eligible.loc[index, "download"]
    dir_name = eligible.loc[index, 'dir_name']
    os.chdir("/ferrero/stan_data/")
    os.chdir(dir_name)

    if isinstance(download_files, list):
        for i in download_files:
            download(i, dir_name)
    else:
        download(download_files, dir_name)


