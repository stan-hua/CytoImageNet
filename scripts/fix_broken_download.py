from describe_dataset import contains_list, str_to_eval
import pandas as pd
import os

home_dir = "/home/stan"
df = pd.read_csv(f"{home_dir}/annotations/datasets_info.csv")
df = df.applymap(str_to_eval)

eligible = df[df.dir_name.str.contains("rec")]

for index in eligible.index:
    download_files = eligible.loc[index, "download"]
    os.chdir("/ferrero/stan_data/")
    os.chdir(eligible.loc[index, 'dir_name'])

    if isinstance(download_files, list):
        for i in download_files:
            os.system(f"wget -c {i}")
    else:
        os.system(f"wget -c {download_files}")
