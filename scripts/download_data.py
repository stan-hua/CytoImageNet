from describe_dataset import contains_list, str_to_eval
import pandas as pd
import os


def download(x: str, name: str) -> None:
    try:
        if "kag" in name:
            os.system(x)
        elif "idr" in name:
            os.system("ascp -TQ -l40m -P 33001 -i "
                      "'/home/stan/.aspera/cli/etc/asperaweb_id_dsa.openssh' "
                      f"{name}@fasp.ebi.ac.uk:. ./")
        else:
            os.system(f"wget {x}")

    except:
        raise Exception(f"{name} failed to download!")


def download_nonexisting(df: pd.DataFrame) -> None:
    data_dir = "/ferrero/stan_data/"

    to_download = df.dir_name.map(lambda x: x not in os.listdir(data_dir))
    print(f"{len(to_download[to_download])} left to download!")

    for i in df.index:
        if df.loc[i, "dir_name"] in os.listdir(data_dir):
            continue
        os.chdir(data_dir)
        os.mkdir(df.loc[i, "dir_name"])
        os.chdir(df.loc[i, "dir_name"])

        if isinstance(df.loc[i, "download"], list):
            for download_cmd in df.loc[i, "download"]:
                download(download_cmd, df.loc[i, "dir_name"])
        else:
            download(download_cmd, df.loc[i, "dir_name"])




if __name__ == "__main__":
    df = pd.read_csv("/home/stan/annotations/datasets_info.csv")
    df.download = df.download.map(contains_list)
    download_nonexisting(df)

    # Remove empty directories
    os.system("find /ferrero/stan_data -type d -empty -delete")

